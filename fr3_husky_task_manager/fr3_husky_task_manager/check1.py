import rclpy
from rclpy.node import Node
from datetime import datetime
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
import pickle
import numpy as np
import cv2

class RealTimeDataSaver(Node):
    def __init__(self):
        super().__init__('realtime_data_saver')
        
        # 1. 파일 설정 (바이너리 쓰기 모드 'ab')
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"pkl_data/realtime_ros_data_{current_time_str}.pkl"
        self.file_handle = open(self.filename, "wb")

        
        
        # 2. 필터링할 관절 이름 목록 정의
        self.target_joints = [
            'left_fr3_joint1', 'left_fr3_joint2', 'left_fr3_joint3', 'left_fr3_joint4', 
            'left_fr3_joint5', 'left_fr3_joint6', 'left_fr3_joint7', 
            'left_fr3_finger_joint1', 'left_fr3_finger_joint2',
            'right_fr3_joint1', 'right_fr3_joint2', 'right_fr3_joint3', 'right_fr3_joint4', 
            'right_fr3_joint5', 'right_fr3_joint6', 'right_fr3_joint7', 
            'right_fr3_finger_joint1', 'right_fr3_finger_joint2'
        ]
        
        # 3. 구독 개시 현재 시간 기록 (실시간 필터링용)
        self.start_time = self.get_clock().now()

        # 4. 기존 토픽 구독 설정 (sensor_msgs)
        self.sub_img_right = self.create_subscription(
            Image, '/mujoco_ros_hardware/right_d435i/color/image_raw', self.image_right_callback, 10)
        self.sub_img_top = self.create_subscription(
            Image, '/mujoco_ros_hardware/top_azure/color/image_raw', self.image_top_callback, 10)
        self.sub_joints = self.create_subscription(
            JointState, '/joint_states', self.joint_states_callback, 10)

        # 5. 신규 PoseStamped 토픽 4종 구독 설정 (geometry_msgs)
        self.pose_topics = [
            '/debug/cur_pose_left',
            '/debug/cur_pose_right',
            '/debug/target_smooth_pose_left',
            '/debug/target_smooth_pose_right'
        ]
        
        self.pose_subs = []
        for topic_name in self.pose_topics:
            sub = self.create_subscription(
                PoseStamped, 
                topic_name, 
                # 람다 함수를 활용하여 콜백에 어떤 토픽에서 들어왔는지 이름을 전달
                lambda msg, t_name=topic_name: self.pose_stamped_callback(msg, t_name), 
                10
            )
            self.pose_subs.append(sub)

        self.get_logger().info(f"실시간 데이터 수집 시작 (Pose 4종 추가 완료) -> {self.filename}")

    def save_to_pickle(self, data):
        """데이터를 pickle 형태로 파일 끝에 추가 저장하는 헬퍼 함수"""
        pickle.dump(data, self.file_handle)

    def center_crop_resize_image_msg(self, msg, target_size=224):
        data = np.frombuffer(msg.data, dtype=np.uint8)

        if msg.encoding in ["rgb8", "bgr8"]:
            img = data.reshape(msg.height, msg.width, 3)
        elif msg.encoding in ["rgba8", "bgra8"]:
            img = data.reshape(msg.height, msg.width, 4)
            img = img[:, :, :3]
        elif msg.encoding in ["mono8", "8UC1"]:
            img = data.reshape(msg.height, msg.width)
        else:
            raise ValueError(f"Unsupported image encoding: {msg.encoding}")

        h, w = img.shape[:2]
        crop_size = min(h, w)
        y0 = (h - crop_size) // 2
        x0 = (w - crop_size) // 2
        img = img[y0:y0 + crop_size, x0:x0 + crop_size]

        interpolation = cv2.INTER_AREA if crop_size > target_size else cv2.INTER_LINEAR
        img = cv2.resize(img, (target_size, target_size), interpolation=interpolation)

        if msg.encoding == "bgr8":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif msg.encoding == "bgra8":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    
    def image_right_callback(self, msg):
        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
        if msg_time > self.start_time:
            img = self.center_crop_resize_image_msg(msg, target_size=224)
            data = {
                'topic': '/mujoco_ros_hardware/right_d435i/color/image_raw',
                'sec': msg.header.stamp.sec,
                'nanosec': msg.header.stamp.nanosec,
                'data': img.tobytes(),
                'height': 224,
                'width': 224,
                'encoding': 'mono8' if img.ndim == 2 else 'rgb8',
                'step': 224 if img.ndim == 2 else 224 * 3
            }
            self.save_to_pickle(data)

    def image_top_callback(self, msg):
        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
        if msg_time > self.start_time:
            img = self.center_crop_resize_image_msg(msg, target_size=224)
            data = {
                'topic': '/mujoco_ros_hardware/top_azure/color/image_raw',
                'sec': msg.header.stamp.sec,
                'nanosec': msg.header.stamp.nanosec,
                'data': img.tobytes(),
                'height': 224,
                'width': 224,
                'encoding': 'mono8' if img.ndim == 2 else 'rgb8',
                'step': 224 if img.ndim == 2 else 224 * 3
            }
            self.save_to_pickle(data)


    def joint_states_callback(self, msg):
        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
        if msg_time > self.start_time:
            joint_map = {name: (pos, vel) for name, pos, vel in zip(msg.name, msg.position, msg.velocity)}
            filtered_positions = []
            filtered_velocities = []
            
            for joint_name in self.target_joints:
                if joint_name in joint_map:
                    pos, vel = joint_map[joint_name]
                    filtered_positions.append(pos)
                    filtered_velocities.append(vel)
            
            data = {
                'topic': '/joint_states',
                'sec': msg.header.stamp.sec,
                'nanosec': msg.header.stamp.nanosec,
                'joint_names': self.target_joints,
                'position': filtered_positions,
                'velocity': filtered_velocities
            }
            self.save_to_pickle(data)

    def pose_stamped_callback(self, msg, topic_name):
        """PoseStamped 토픽 4종을 통합 처리하는 공용 콜백 함수"""
        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
        if msg_time > self.start_time:
            p = msg.pose.position
            o = msg.pose.orientation
            
            data = {
                'topic': topic_name,
                'sec': msg.header.stamp.sec,
                'nanosec': msg.header.stamp.nanosec,
                'frame_id': msg.header.frame_id,
                # 위치 및 쿼터니언 데이터 분리 파싱
                'position': {'x': p.x, 'y': p.y, 'z': p.z},
                'orientation': {'x': o.x, 'y': o.y, 'z': o.z, 'w': o.w}
            }
            self.save_to_pickle(data)

    def destroy_node(self):
        if hasattr(self, 'file_handle') and not self.file_handle.closed:
            self.file_handle.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RealTimeDataSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
