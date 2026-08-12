import rclpy
from rclpy.node import Node
from datetime import datetime
from sensor_msgs.msg import JointState, CompressedImage 
from geometry_msgs.msg import PoseStamped
import pickle
import time
import numpy as np
import cv2


class RealTimeDataSaver(Node):
    def __init__(self):
        super().__init__('realtime_data_saver')
        
        # 1. 파일 설정 (바이너리 쓰기 모드 'wb')
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"pkl_data/realtime_ros_data_{current_time_str}.pkl"
        self.file_handle = open(self.filename, "wb")

        # Hz (fps) 설정 및 주기 계산 (30 FPS)
        self.save_rate = 30.0
        self.save_period = 1.0 / self.save_rate

        self.image_topic_name = '/camera/camera/color/image_raw/compressed'

        self.last_save_time = {
            self.image_topic_name: 0.0,
            '/joint_states': 0.0,
            '/debug/cur_pose_left': 0.0,
            '/debug/cur_pose_right': 0.0,
            '/debug/target_smooth_pose_left': 0.0,
            '/debug/target_smooth_pose_right': 0.0,
        }

        self.target_image_size = 84

        # 2. 필터링할 관절 이름 목록 정의
        self.target_joints = [
            'left_fr3_joint1', 'left_fr3_joint2', 'left_fr3_joint3', 'left_fr3_joint4', 
            'left_fr3_joint5', 'left_fr3_joint6', 'left_fr3_joint7', 
            'left_fr3_finger_joint1', 'left_fr3_finger_joint2',
            'right_fr3_joint1', 'right_fr3_joint2', 'right_fr3_joint3', 'right_fr3_joint4', 
            'right_fr3_joint5', 'right_fr3_joint6', 'right_fr3_joint7', 
            'right_fr3_finger_joint1', 'right_fr3_finger_joint2'
        ]
    
        self.start_time = self.get_clock().now()

        self.sub_img_top = self.create_subscription(
            CompressedImage, self.image_topic_name, self.image_top_callback, 10)
        self.sub_joints = self.create_subscription(
            JointState, '/joint_states', self.joint_states_callback, 10)

        # 5. PoseStamped 토픽 4종 구독 설정
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
                lambda msg, t_name=topic_name: self.pose_stamped_callback(msg, t_name), 
                10
            )
            self.pose_subs.append(sub)

        self.get_logger().info(f"실시간 데이터 수집 시작 (컬러 압축 이미지 모드) -> {self.filename}")

    def should_save(self, topic_name):
        now = time.monotonic()
        if now - self.last_save_time[topic_name] < self.save_period:
            return False
        self.last_save_time[topic_name] = now
        return True

    def save_to_pickle(self, data):
        pickle.dump(data, self.file_handle)

    def process_compressed_image_msg(self, msg, target_size=84):
        """JPEG/PNG로 압축된 바이너리 데이터를 RGB 넘파이 배열로 복원 및 크롭 가공"""
        # 1. 압축 바이너리 배열을 OpenCV 이미지로 디코딩 (기본 BGR 컬러 포맷으로 해제됨)
        np_arr = np.frombuffer(msg.data, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise ValueError("Compressed image decoding failed.")

        # 2. 중앙 크롭 처리
        h, w = img_bgr.shape[:2]
        crop_size = min(h, w)
        y0 = (h - crop_size) // 2
        x0 = (w - crop_size) // 2
        img = img_bgr[y0:y0 + crop_size, x0:x0 + crop_size]

        # 3. 84x84 리사이즈
        interpolation = cv2.INTER_AREA if crop_size > target_size else cv2.INTER_LINEAR
        img = cv2.resize(img, (target_size, target_size), interpolation=interpolation)

        # 4. 학습 네트워크 관례에 맞춰 BGR에서 RGB로 색상 채널 전환
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb

    def image_top_callback(self, msg):
        if not self.should_save(self.image_topic_name): return
        
        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
        if msg_time > self.start_time:
            try:
                # 변경된 컬러 압축 이미지 가공 함수 호출
                img = self.process_compressed_image_msg(msg, target_size=self.target_image_size)
                
                # robomimic 변환 스크립트 규격 구조 유지
                data = {
                    'topic': self.image_topic_name,
                    'sec': msg.header.stamp.sec,
                    'nanosec': msg.header.stamp.nanosec,
                    'data': img.tobytes(),
                    'height': self.target_image_size,
                    'width': self.target_image_size,
                    'encoding': 'rgb8', 
                    'step': self.target_image_size * 3  # 채널이 3개이므로 너비 * 3
                }
                self.save_to_pickle(data)
            except Exception as e:
                self.get_logger().error(f"Top Compressed Color Image 저장 오류: {e}")

    def joint_states_callback(self, msg):
        topic_name = '/joint_states'
        if not self.should_save(topic_name): return

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
                'topic': topic_name,
                'sec': msg.header.stamp.sec,
                'nanosec': msg.header.stamp.nanosec,
                'joint_names': self.target_joints,
                'position': filtered_positions,
                'velocity': filtered_velocities
            }
            self.save_to_pickle(data)

    def pose_stamped_callback(self, msg, topic_name):
        if not self.should_save(topic_name): return

        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
        if msg_time > self.start_time:
            data = {
                'topic': topic_name,
                'sec': msg.header.stamp.sec,
                'nanosec': msg.header.stamp.nanosec,
                'position': {
                    'x': msg.pose.position.x,
                    'y': msg.pose.position.y,
                    'z': msg.pose.position.z
                },
                'orientation': {
                    'x': msg.pose.orientation.x,
                    'y': msg.pose.orientation.y,
                    'z': msg.pose.orientation.z,
                    'w': msg.pose.orientation.w
                }
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
