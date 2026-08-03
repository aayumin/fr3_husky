import json
import pickle
from pathlib import Path
import re

import cv2
import h5py
import numpy as np

# [수정] 여러 pkl 파일이 모여있는 폴더 경로 지정
PKL_DIR = "./"  # 현재 폴더 기준 (필요시 절대 경로로 변경 가능)
OUT_HDF5_PATH = "/home/yuminlim/project/robomimic/datasets/__test__/test_data.hdf5"

AGENTVIEW_TOPIC = "/mujoco_ros_hardware/top_azure/color/image_raw"
EYE_IN_HAND_TOPIC = "/mujoco_ros_hardware/right_d435i/color/image_raw"

EEF_POSE_TOPIC = "/debug/cur_pose_left"
TARGET_EEF_POSE_TOPIC = "/debug/target_smooth_pose_left"
JOINT_TOPIC = "/joint_states"

LEFT_ARM_JOINTS = [
    "left_fr3_joint1",
    "left_fr3_joint2",
    "left_fr3_joint3",
    "left_fr3_joint4",
    "left_fr3_joint5",
    "left_fr3_joint6",
    "left_fr3_joint7",
]

LEFT_GRIPPER_JOINTS = [
    "left_fr3_finger_joint1",
    "left_fr3_finger_joint2",
]

# IMAGE_SIZE = 224
IMAGE_SIZE = 84
# CONTROL_DT = 0.1  
CONTROL_DT = 0.05  # 수정 후 (FPS = 20 세팅)
NORMALIZE_ACTIONS = False
ACTION_SCALE = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)




robomimic_compat_env_args = {
    "env_name": "NutAssemblySquare",  # 스크립트가 인식할 수 있는 공식 시뮬레이션 이름으로 매핑
    "env_version": "1.5.1",
    "type": 1,                        # 1: robosuite 기반 환경 타입 명시
    "env_kwargs": {
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "use_object_obs": True,
        "use_camera_obs": True,
        "control_freq": 20,           # 👈 FPS=20 (CONTROL_DT = 0.05) 규격 동기화
        "controller_configs": {
            "type": "BASIC",
            "body_parts": {
                "right": {            # robomimic/robosuite의 기본 단일 팔 칭호는 보통 'right' 또는 'robot0' 기반입니다.
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
                    "gripper": {"type": "GRIP"}
                }
            }
        },
        "robots": ["Panda"], 
        "camera_depths": False,
        # "camera_heights": 224,
        # "camera_widths": 224,
        "camera_heights": 84,
        "camera_widths": 84,
        "lite_physics": False,
        "reward_shaping": False,
        "camera_names": ["agentview", "robot0_eye_in_hand"], 
        "render_gpu_device_id": 0
    }
}



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
    for r in records:
        topics.setdefault(r["topic"], []).append(r)
    for topic in topics:
        topics[topic].sort(key=stamp_to_sec)
    return topics


def nearest_record(records, t):
    if len(records) == 0:
        return None
    times = np.array([stamp_to_sec(r) for r in records])
    idx = int(np.argmin(np.abs(times - t)))
    return records[idx]


def decode_image(record, out_size=84):
    h, w, enc = record["height"], record["width"], record["encoding"]
    data = np.frombuffer(record["data"], dtype=np.uint8)

    if enc in ["rgb8", "bgr8"]:
        img = data.reshape(h, w, 3)
        if enc == "bgr8":
            img = img[:, :, ::-1]
    elif enc in ["rgba8", "bgra8"]:
        img = data.reshape(h, w, 4)[:, :, :3]
        if enc == "bgra8":
            img = img[:, :, ::-1]
    elif enc in ["mono8", "8UC1"]:
        img = data.reshape(h, w)
        img = np.repeat(img[:, :, None], 3, axis=2)
    else:
        raise ValueError(f"Unsupported image encoding: {enc}")

    if img.shape[0] != out_size or img.shape[1] != out_size:
        img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)

    return img.astype(np.uint8)


def pose_to_pos_quat(record):
    p = record["position"]
    q = record["orientation"]
    pos = np.array([p["x"], p["y"], p["z"]], dtype=np.float32)
    quat = np.array([q["x"], q["y"], q["z"], q["w"]], dtype=np.float32)
    quat = quat / max(np.linalg.norm(quat), 1e-8)
    return pos, quat


def joint_record_to_arrays(record):
    names = record["joint_names"]
    pos_map = {n: v for n, v in zip(names, record["position"])}
    vel_map = {n: v for n, v in zip(names, record["velocity"])}

    joint_pos = np.array([pos_map[n] for n in LEFT_ARM_JOINTS], dtype=np.float32)
    joint_vel = np.array([vel_map[n] for n in LEFT_ARM_JOINTS], dtype=np.float32)
    gripper_qpos = np.array([pos_map[n] for n in LEFT_GRIPPER_JOINTS], dtype=np.float32)
    gripper_qvel = np.array([vel_map[n] for n in LEFT_GRIPPER_JOINTS], dtype=np.float32)

    return joint_pos, joint_vel, gripper_qpos, gripper_qvel


def make_sample_times(joint_records, dt):
    t0 = stamp_to_sec(joint_records[0])
    t1 = stamp_to_sec(joint_records[-1])
    return np.arange(t0, t1, dt, dtype=np.float64)

def build_demo_arrays(topics):
    joint_records = topics.get(JOINT_TOPIC, [])
    agentview_records = topics.get(AGENTVIEW_TOPIC, [])
    eye_records = topics.get(EYE_IN_HAND_TOPIC, [])
    
    # [수정] 4종의 Pose 토픽 중 필요한 토픽들을 명확히 가져옴
    eef_cur_records = topics.get("/debug/cur_pose_left", [])
    eef_target_records = topics.get("/debug/target_smooth_pose_left", [])

    if not joint_records: raise RuntimeError(f"Missing topic: {JOINT_TOPIC}")
    if not agentview_records: raise RuntimeError(f"Missing topic: {AGENTVIEW_TOPIC}")
    if not eye_records: raise RuntimeError(f"Missing topic: {EYE_IN_HAND_TOPIC}")
    if not eef_cur_records: raise RuntimeError("Missing topic: /debug/cur_pose_left")
    if not eef_target_records: raise RuntimeError("Missing topic: /debug/target_smooth_pose_left")

    # 관절 상태 타임스탬프 기준으로 샘플링 타임 생성
    sample_times = make_sample_times(joint_records, CONTROL_DT)

    agentview_images = []
    eye_in_hand_images = []
    gripper_qpos = []
    gripper_qvel = []
    joint_pos = []
    joint_vel = []
    
    # EEF Observation 및 Action 저장을 위한 리스트
    eef_pos = []
    eef_quat = []
    action_list = []

    for t in sample_times:
        joint_r = nearest_record(joint_records, t)
        agentview_r = nearest_record(agentview_records, t)
        eye_r = nearest_record(eye_records, t)
        
        # 1. 현재 로봇 상태 관측값 (Observation) 동기화 추출
        eef_cur_r = nearest_record(eef_cur_records, t)
        
        # 2. 로봇 제어 명령값 (Action) 동기화 추출
        eef_target_r = nearest_record(eef_target_records, t)

        jp, jv, gq, gv = joint_record_to_arrays(joint_r)
        
        # 현재 Pose 파싱 -> Observation 데이터로 사용
        ep, eq = pose_to_pos_quat(eef_cur_r)
        
        # 타겟 Pose 파싱 -> Action 데이터로 사용
        ap, aq = pose_to_pos_quat(eef_target_r)
        
        # Diffusion Policy의 카테시안 액션 표준 포맷: [pos(3), quat(4)] = 총 7차원
        action_step = np.concatenate([ap, aq], axis=0) 

        joint_pos.append(jp)
        joint_vel.append(jv)
        gripper_qpos.append(gq)
        gripper_qvel.append(gv)
        eef_pos.append(ep)
        eef_quat.append(eq)
        action_list.append(action_step)
        
        agentview_images.append(decode_image(agentview_r, IMAGE_SIZE))
        eye_in_hand_images.append(decode_image(eye_r, IMAGE_SIZE))

    # numpy 배열 변환
    actions = np.asarray(action_list, dtype=np.float32) # 이 부분이 타겟 명령값 기반의 진짜 action이 됩니다.
    
    joint_pos = np.asarray(joint_pos, dtype=np.float32)
    joint_vel = np.asarray(joint_vel, dtype=np.float32)
    gripper_qpos = np.asarray(gripper_qpos, dtype=np.float32)
    gripper_qvel = np.asarray(gripper_qvel, dtype=np.float32)
    eef_pos = np.asarray(eef_pos, dtype=np.float32)
    eef_quat = np.asarray(eef_quat, dtype=np.float32)
    agentview_images = np.asarray(agentview_images, dtype=np.uint8)
    eye_in_hand_images = np.asarray(eye_in_hand_images, dtype=np.uint8)

    if NORMALIZE_ACTIONS:
        actions = np.clip(actions / ACTION_SCALE, -1.0, 1.0).astype(np.float32)

    dones = np.zeros((len(sample_times),), dtype=np.int64)
    dones[-1] = 1

    return {
        "actions": actions, # 동기화된 [target_pos, target_quat] 배열주입
        "dones": dones,
        "obs": {
            "agentview_image": agentview_images,
            "robot0_eye_in_hand_image": eye_in_hand_images,
            "robot0_eef_pos": eef_pos,      # 현재 상태의 eef pos
            "robot0_eef_quat": eef_quat,    # 현재 상태의 eef quat
            "robot0_gripper_qpos": gripper_qpos,
            "robot0_gripper_qvel": gripper_qvel,
            "robot0_joint_pos": joint_pos,
            "robot0_joint_vel": joint_vel,
        },
    }


def write_robomimic_hdf5_multi(out_path, demo_list):
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
            
            print(f"-> [{demo_name}] 변환 완료: 스텝 수={num_samples}, 액션 Shape={demo['actions'].shape}")

        # 글로벌 공통 메타데이터 주입
        data_grp.attrs["total"] = total_samples
        
        # ====================================================
        # [수정] robomimic train 파이프라인 우회용 가상 규격 주입
        # ====================================================
        compat_env_args = {
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
                            "gripper": {"type": "GRIP"}
                        }
                    }
                },
                "robots": ["Franka"],
                "camera_depths": False,
                "camera_heights": int(IMAGE_SIZE),
                "camera_widths": int(IMAGE_SIZE),
                "lite_physics": False,
                "reward_shaping": False,
                "camera_names": ["agentview", "robot0_eye_in_hand"],
                "render_gpu_device_id": 0
            }
        }
        data_grp.attrs["env_args"] = json.dumps(compat_env_args)
        # ====================================================

        str_dtype = h5py.string_dtype(encoding="utf-8")
        num_demos = len(demo_names)
        
        if num_demos >= 2:
            split_idx = max(1, int(num_demos * 0.9))
            train_demos = demo_names[:split_idx]
            valid_demos = demo_names[split_idx:]
        else:
            train_demos = demo_names
            valid_demos = demo_names

        mask_grp.create_dataset("train", data=np.array(train_demos, dtype=object), dtype=str_dtype)
        mask_grp.create_dataset("valid", data=np.array(valid_demos, dtype=object), dtype=str_dtype)

    print("\n==============================================")
    print(f"🎉 가상 메타데이터 규격 포함 멀티 HDF5 저장 완료: {out_path}")
    print(f"총 에피소드 수: {num_demos} 개")
    print(f"총 샘플 타임스텝 수: {total_samples} 단계 (설정 주종: {1/CONTROL_DT}Hz)")
    print(f"학습용 에피소드 (Train): {train_demos}")
    print(f"검증용 에피소드 (Valid): {valid_demos}")
    print("==============================================")




def main():

    pkl_folder = Path(PKL_DIR)
    
    # 1. 정규표현식을 통해 'realtime_ros_data_*.pkl' 형태를 가진 파일들을 전부 긁어와 정렬
    pkl_files = sorted([p for p in pkl_folder.glob("realtime_ros_data_*.pkl")])
    
    if not pkl_files:
        print(f"❌ 에러: [{pkl_folder.resolve()}] 경로에 'realtime_ros_data_*.pkl' 파일이 하나도 없습니다.")
        exit(1)
        
    print(f"📚 수집된 에피소드 파일 발견: {len(pkl_files)}개 목록을 순차 변환 시작합니다...")

    all_demos = []
    
    # 2. 각 pkl 파일들을 하나씩 해체해서 robomimic 배열 형식으로 압축
    for pkl_file in pkl_files:
        print(f"\n[읽는 중] 파일명: {pkl_file.name}")
        try:
            records = load_pickle_stream(pkl_file)
            topics = split_by_topic(records)
            demo = build_demo_arrays(topics)
            all_demos.append(demo)
        except Exception as e:
            print(f"⚠️ 경고: {pkl_file.name} 처리 도중 오류가 발생하여 스킵합니다. 에러: {e}")
            continue

    # 3. 누적된 에피소드 리스트들을 모아 멀티 구조 hdf5 파일 빌드
    if all_demos:
        write_robomimic_hdf5_multi(OUT_HDF5_PATH, all_demos)
    else:
        print("❌ 유효하게 처리된 에피소드 배열이 하나도 없어 변환을 취소합니다.")



if __name__ == "__main__":
    main()