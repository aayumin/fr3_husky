#!/usr/bin/env python3

import argparse
import json
import pickle
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm


PKL_DIR = "./"
OUT_HDF5_PATH = "/home/yuminlim/project/robomimic/datasets/__test__/test_data.hdf5"

AGENTVIEW_TOPIC = "/camera/camera/color/image_raw/compressed"
EYE_IN_HAND_TOPIC = "/mujoco_ros_hardware/right_d435i/color/image_raw"
JOINT_TOPIC = "/joint_states"

EEF_CUR_TOPICS = {
    "left": "/debug/cur_pose_left",
    "right": "/debug/cur_pose_right",
}

EEF_TARGET_TOPICS = {
    "left": "/debug/target_smooth_pose_left",
    "right": "/debug/target_smooth_pose_right",
}

ARM_JOINTS = {
    "left": [
        "left_fr3_joint1",
        "left_fr3_joint2",
        "left_fr3_joint3",
        "left_fr3_joint4",
        "left_fr3_joint5",
        "left_fr3_joint6",
        "left_fr3_joint7",
    ],
    "right": [
        "right_fr3_joint1",
        "right_fr3_joint2",
        "right_fr3_joint3",
        "right_fr3_joint4",
        "right_fr3_joint5",
        "right_fr3_joint6",
        "right_fr3_joint7",
    ],
}

GRIPPER_JOINTS = {
    "left": ["left_fr3_finger_joint1", "left_fr3_finger_joint2"],
    "right": ["right_fr3_finger_joint1", "right_fr3_finger_joint2"],
}

IMAGE_SIZE = 84
CONTROL_DT = 0.05
NORMALIZE_ACTIONS = False

SINGLE_ARM_ACTION_SCALE = np.array(
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    dtype=np.float32,
)


def get_selected_arms(arm_mode):
    if arm_mode == "dual":
        return ["left", "right"]
    return [arm_mode]


def load_pickle_stream(path):
    records = []

    with open(path, "rb") as f:
        while True:
            try:
                records.append(pickle.load(f))
            except EOFError:
                break

    return records


def stamp_to_sec(record):
    return float(record["sec"]) + float(record["nanosec"]) * 1e-9


def split_by_topic(records):
    topics = {}

    for record in tqdm(records):
        topics.setdefault(record["topic"], []).append(record)

    for topic in topics:
        topics[topic].sort(key=stamp_to_sec)

    return topics


def nearest_record(records, t):
    if len(records) == 0:
        return None

    times = np.array([stamp_to_sec(record) for record in records], dtype=np.float64)
    idx = int(np.argmin(np.abs(times - t)))

    return records[idx]


def decode_image(record, out_size=84):
    h = int(record["height"])
    w = int(record["width"])
    enc = record["encoding"]
    data = np.frombuffer(record["data"], dtype=np.uint8)

    if enc in ("rgb8", "bgr8"):
        img = data.reshape(h, w, 3)
        if enc == "bgr8":
            img = img[:, :, ::-1]

    elif enc in ("rgba8", "bgra8"):
        img = data.reshape(h, w, 4)[:, :, :3]
        if enc == "bgra8":
            img = img[:, :, ::-1]

    elif enc in ("mono8", "8UC1"):
        img = data.reshape(h, w)
        img = np.repeat(img[:, :, None], 3, axis=2)

    else:
        raise ValueError(f"Unsupported image encoding: {enc}")

    if img.shape[0] != out_size or img.shape[1] != out_size:
        img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)

    return np.ascontiguousarray(img, dtype=np.uint8)


def pose_to_pos_quat(record):
    p = record["position"]
    q = record["orientation"]

    pos = np.array([p["x"], p["y"], p["z"]], dtype=np.float32)
    quat = np.array([q["x"], q["y"], q["z"], q["w"]], dtype=np.float32)

    quat /= max(np.linalg.norm(quat), 1e-8)

    return pos, quat

def quat_xyzw_to_rotvec(quat, eps=1e-8):
    """
    Convert quaternion [x, y, z, w] to rotation vector axis * angle.

    Returns:
        rotvec: np.ndarray, shape [3]
    """
    quat = np.asarray(quat, dtype=np.float32)
    quat = quat / max(np.linalg.norm(quat), eps)

    q_xyz = quat[:3]
    q_w = float(quat[3])

    # Use a canonical hemisphere to reduce sign discontinuity.
    if q_w < 0.0:
        q_xyz = -q_xyz
        q_w = -q_w

    q_w = np.clip(q_w, -1.0, 1.0)

    sin_half = np.linalg.norm(q_xyz)

    if sin_half < eps:
        return np.zeros(3, dtype=np.float32)

    angle = 2.0 * np.arctan2(sin_half, q_w)
    axis = q_xyz / sin_half
    rotvec = axis * angle

    return rotvec.astype(np.float32)

def gripper_qpos_to_action(gripper_qpos):
    """
    Convert 2 finger joint positions into one scalar gripper action.

    For absolute-action real-world data, this keeps the gripper in its native
    physical scale. Action normalization in the training pipeline can handle
    the magnitude later.

    Args:
        gripper_qpos: np.ndarray, shape [2] for single arm

    Returns:
        gripper_action: np.ndarray, shape [1]
    """
    gripper_qpos = np.asarray(gripper_qpos, dtype=np.float32)

    if gripper_qpos.size == 0:
        return np.zeros((1,), dtype=np.float32)

    # Mean is stable when two finger joints move symmetrically.
    gripper_value = np.mean(gripper_qpos)

    return np.array([gripper_value], dtype=np.float32)


def extract_joint_arrays(record, arm):
    names = record["joint_names"]

    pos_map = {name: value for name, value in zip(names, record["position"])}
    vel_map = {name: value for name, value in zip(names, record["velocity"])}

    arm_joints = ARM_JOINTS[arm]
    gripper_joints = GRIPPER_JOINTS[arm]

    missing_pos = [name for name in arm_joints + gripper_joints if name not in pos_map]
    missing_vel = [name for name in arm_joints + gripper_joints if name not in vel_map]

    if missing_pos:
        raise RuntimeError(f"Missing joint positions for {arm}: {missing_pos}")

    if missing_vel:
        raise RuntimeError(f"Missing joint velocities for {arm}: {missing_vel}")

    joint_pos = np.array([pos_map[name] for name in arm_joints], dtype=np.float32)
    joint_vel = np.array([vel_map[name] for name in arm_joints], dtype=np.float32)
    gripper_qpos = np.array([pos_map[name] for name in gripper_joints], dtype=np.float32)
    gripper_qvel = np.array([vel_map[name] for name in gripper_joints], dtype=np.float32)

    return joint_pos, joint_vel, gripper_qpos, gripper_qvel


def joint_record_to_arrays(record, arm_mode):
    selected_arms = get_selected_arms(arm_mode)

    joint_pos_list = []
    joint_vel_list = []
    gripper_qpos_list = []
    gripper_qvel_list = []

    for arm in selected_arms:
        jp, jv, gq, gv = extract_joint_arrays(record, arm)

        joint_pos_list.append(jp)
        joint_vel_list.append(jv)
        gripper_qpos_list.append(gq)
        gripper_qvel_list.append(gv)

    joint_pos = np.concatenate(joint_pos_list, axis=0)
    joint_vel = np.concatenate(joint_vel_list, axis=0)
    gripper_qpos = np.concatenate(gripper_qpos_list, axis=0)
    gripper_qvel = np.concatenate(gripper_qvel_list, axis=0)

    return joint_pos, joint_vel, gripper_qpos, gripper_qvel


def make_sample_times(joint_records, dt):
    t0 = stamp_to_sec(joint_records[0])
    t1 = stamp_to_sec(joint_records[-1])
    return np.arange(t0, t1, dt, dtype=np.float64)


def check_required_topics(topics, arm_mode):
    # required_topics = [JOINT_TOPIC, AGENTVIEW_TOPIC, EYE_IN_HAND_TOPIC]
    required_topics = [JOINT_TOPIC, AGENTVIEW_TOPIC]

    for arm in get_selected_arms(arm_mode):
        required_topics.append(EEF_CUR_TOPICS[arm])
        required_topics.append(EEF_TARGET_TOPICS[arm])

    missing_topics = [topic for topic in required_topics if not topics.get(topic)]

    if missing_topics:
        raise RuntimeError(f"Missing topics: {missing_topics}")


def build_demo_arrays(topics, arm_mode):
    check_required_topics(topics, arm_mode)

    selected_arms = get_selected_arms(arm_mode)

    joint_records = topics[JOINT_TOPIC]
    agentview_records = topics[AGENTVIEW_TOPIC]
    # eye_records = topics[EYE_IN_HAND_TOPIC]

    eef_cur_records = {arm: topics[EEF_CUR_TOPICS[arm]] for arm in selected_arms}
    eef_target_records = {arm: topics[EEF_TARGET_TOPICS[arm]] for arm in selected_arms}

    sample_times = make_sample_times(joint_records, CONTROL_DT)

    agentview_images = []
    # eye_in_hand_images = []

    joint_pos = []
    joint_vel = []
    gripper_qpos = []
    gripper_qvel = []

    eef_pos = []
    eef_quat = []
    action_list = []

    for t in tqdm(sample_times):
        joint_r = nearest_record(joint_records, t)
        agentview_r = nearest_record(agentview_records, t)
        # eye_r = nearest_record(eye_records, t)

        jp, jv, gq, gv = joint_record_to_arrays(joint_r, arm_mode)

        current_pos_list = []
        current_quat_list = []
        target_action_list = []

        for arm in selected_arms:
            eef_cur_r = nearest_record(eef_cur_records[arm], t)
            eef_target_r = nearest_record(eef_target_records[arm], t)

            current_pos, current_quat = pose_to_pos_quat(eef_cur_r)
            target_pos, target_quat = pose_to_pos_quat(eef_target_r)
            target_rotvec = quat_xyzw_to_rotvec(target_quat)

            current_pos_list.append(current_pos)
            current_quat_list.append(current_quat)

            if arm_mode == "dual":
                arm_idx = selected_arms.index(arm)
                arm_gripper_qpos = gq[2 * arm_idx : 2 * arm_idx + 2]
            else:
                arm_gripper_qpos = gq

            target_gripper = gripper_qpos_to_action(arm_gripper_qpos)
            arm_action = np.concatenate([target_pos, target_rotvec, target_gripper], axis=0).astype(np.float32)
            target_action_list.append(arm_action)

        current_eef_pos = np.concatenate(current_pos_list, axis=0)
        current_eef_quat = np.concatenate(current_quat_list, axis=0)
        action_step = np.concatenate(target_action_list, axis=0)

        joint_pos.append(jp)
        joint_vel.append(jv)
        gripper_qpos.append(gq)
        gripper_qvel.append(gv)

        eef_pos.append(current_eef_pos)
        eef_quat.append(current_eef_quat)
        action_list.append(action_step)

        agentview_images.append(decode_image(agentview_r, IMAGE_SIZE))
        # eye_in_hand_images.append(decode_image(eye_r, IMAGE_SIZE))

    actions = np.asarray(action_list, dtype=np.float32)
    joint_pos = np.asarray(joint_pos, dtype=np.float32)
    joint_vel = np.asarray(joint_vel, dtype=np.float32)
    gripper_qpos = np.asarray(gripper_qpos, dtype=np.float32)
    gripper_qvel = np.asarray(gripper_qvel, dtype=np.float32)
    eef_pos = np.asarray(eef_pos, dtype=np.float32)
    eef_quat = np.asarray(eef_quat, dtype=np.float32)
    agentview_images = np.asarray(agentview_images, dtype=np.uint8)
    # eye_in_hand_images = np.asarray(eye_in_hand_images, dtype=np.uint8)

    if NORMALIZE_ACTIONS:
        action_scale = np.tile(SINGLE_ARM_ACTION_SCALE, len(selected_arms))
        actions = np.clip(actions / action_scale, -1.0, 1.0).astype(np.float32)

    dones = np.zeros((len(sample_times),), dtype=np.int64)

    if len(dones) > 0:
        dones[-1] = 1

    return {
        "actions": actions,
        "dones": dones,
        "obs": {
            "agentview_image": agentview_images,
            # "robot0_eye_in_hand_image": eye_in_hand_images,
            "robot0_eye_in_hand_image": agentview_images,
            "robot0_eef_pos": eef_pos,
            "robot0_eef_quat": eef_quat,
            "robot0_gripper_qpos": gripper_qpos,
            "robot0_gripper_qvel": gripper_qvel,
            "robot0_joint_pos": joint_pos,
            "robot0_joint_vel": joint_vel,
        },
    }


def make_compat_env_args():
    return {
        "env_name": "NutAssemblySquare",
        "env_version": "1.5.1",
        "type": 1,
        "env_kwargs": {
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "ignore_done": True,
            "use_object_obs": True,
            "use_camera_obs": True,
            "control_freq": 20,
            "controller_configs": {
                "type": "BASIC",
                "body_parts": {
                    "right": {
                        "type": "OSC_POSE",
                        "input_max": 1,
                        "input_min": -1,
                        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                        "kp": 150,
                        "damping": 1,
                        "impedance_mode": "fixed",
                        "kp_limits": [0, 300],
                        "damping_limits": [0, 10],
                        "position_limits": None,
                        "orientation_limits": None,
                        "uncouple_pos_ori": True,
                        "control_delta": True,
                        "interpolation": None,
                        "ramp_ratio": 0.2,
                        "input_ref_frame": "world",
                        "gripper": {"type": "GRIP"},
                    }
                },
            },
            "robots": ["Panda"],
            "camera_depths": False,
            "camera_heights": int(IMAGE_SIZE),
            "camera_widths": int(IMAGE_SIZE),
            "lite_physics": False,
            "reward_shaping": False,
            "camera_names": ["agentview", "robot0_eye_in_hand"],
            "render_gpu_device_id": 0,
        },
    }


def write_robomimic_hdf5_multi(out_path, demo_list, arm_mode):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        out_path.unlink()

    with h5py.File(out_path, "w") as f:
        data_grp = f.create_group("data")
        mask_grp = f.create_group("mask")

        total_samples = 0
        demo_names = []

        for idx, demo in enumerate(demo_list):
            demo_name = f"demo_{idx}"
            demo_names.append(demo_name)

            demo_grp = data_grp.create_group(demo_name)
            obs_grp = demo_grp.create_group("obs")

            demo_grp.create_dataset("actions", data=demo["actions"])
            demo_grp.create_dataset("dones", data=demo["dones"])

            for key, value in demo["obs"].items():
                obs_grp.create_dataset(key, data=value)

            num_samples = demo["actions"].shape[0]
            demo_grp.attrs["num_samples"] = num_samples
            total_samples += num_samples

            print(f"[{demo_name}] steps={num_samples}, actions={demo['actions'].shape}")

        data_grp.attrs["total"] = total_samples
        data_grp.attrs["arm"] = arm_mode
        data_grp.attrs["env_args"] = json.dumps(make_compat_env_args())

        str_dtype = h5py.string_dtype(encoding="utf-8")
        num_demos = len(demo_names)

        # if num_demos >= 2:
        #     split_idx = max(1, int(num_demos * 0.9))
        #     train_demos = demo_names[:split_idx]
        #     valid_demos = demo_names[split_idx:]
        # else:
        #     train_demos = demo_names
        #     valid_demos = demo_names
        train_demos = demo_names
        valid_demos = demo_names

        mask_grp.create_dataset("train", data=np.array(train_demos, dtype=object), dtype=str_dtype)
        mask_grp.create_dataset("valid", data=np.array(valid_demos, dtype=object), dtype=str_dtype)

    print()
    print("==============================================")
    print(f"Saved: {out_path}")
    print(f"Arm mode: {arm_mode}")
    print(f"Total demos: {num_demos}")
    print(f"Total samples: {total_samples}")
    print(f"Control frequency: {1.0 / CONTROL_DT:.1f} Hz")
    print(f"Train demos: {train_demos}")
    print(f"Valid demos: {valid_demos}")

    if demo_list:
        first_demo = demo_list[0]
        print(f"Action shape: {first_demo['actions'].shape}")

        for key, value in first_demo["obs"].items():
            print(f"{key}: {value.shape}")

    print("==============================================")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--arm", choices=["left", "right", "dual"], required=True, help="Robot arm mode")
    parser.add_argument("--pkl_dir", type=str, default=PKL_DIR)
    parser.add_argument("--output", type=str, default=OUT_HDF5_PATH)

    return parser.parse_args()


def main():
    args = parse_args()

    pkl_folder = Path(args.pkl_dir)
    pkl_files = sorted(pkl_folder.glob("pkl_data/realtime_ros_data_*.pkl"))

    if not pkl_files:
        print(f"No pkl files found in: {pkl_folder.resolve() / 'pkl_data'}")
        return

    print(f"Arm mode: {args.arm}")
    print(f"Found {len(pkl_files)} episodes")

    all_demos = []

    for pkl_file in pkl_files:
        print(f"\nProcessing: {pkl_file.name}")

        try:
            records = load_pickle_stream(pkl_file)
            topics = split_by_topic(records)
            demo = build_demo_arrays(topics, args.arm)
            all_demos.append(demo)

        except Exception as error:
            print(f"Skip {pkl_file.name}: {error}")

    if not all_demos:
        print("No valid episodes were converted.")
        return

    write_robomimic_hdf5_multi(args.output, all_demos, args.arm)


if __name__ == "__main__":
    main()
