import rclpy
from rclpy.node import Node
from datetime import datetime
from sensor_msgs.msg import JointState, CompressedImage
from geometry_msgs.msg import PoseStamped
import pickle
import time
import sys

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

        # 2. 필터링할 관절 이름 목록 정의
        self.target_joints = [
            'left_fr3_joint1', 'left_fr3_joint2', 'left_fr3_joint3', 'left_fr3_joint4', 
            'left_fr3_joint5', 'left_fr3_joint6', 'left_fr3_joint7', 
            'left_fr3_finger_joint1', 'left_fr3_finger_joint2',
            'right_fr3_joint1', 'right_fr3_joint2', 'right_fr3_joint3', 'right_fr3_joint4', 
            'right_fr3_joint5', 'right_fr3_joint6', 'right_fr3_joint7', 
            'right_fr3_finger_joint1', 'right_fr3_finger_joint2'
        ]
        
        # 3. 구독 개시 현재 시간 기록 (과거 메시지 무시용)
        self.start_time = self.get_clock().now()

        # 4. 토픽 구독 설정 (CompressedImage 타입 적용)
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

        self.get_logger().info(f"실시간 데이터 수집 시작 (압축 이미지 원본 바이트 저장 모드) -> {self.filename}")

    def should_save(self, topic_name):
        now = time.monotonic()
        if now - self.last_save_time[topic_name] < self.save_period:
            return False
        self.last_save_time[topic_name] = now
        return True

    def save_to_pickle(self, data):
        pickle.dump(data, self.file_handle)

    def image_top_callback(self, msg):
        """💡 터미널 도배 print 문 제거 및 정상적인 수집 프로세스 유지"""
        if not self.should_save(self.image_topic_name): return
        
        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
        if msg_time > self.start_time:
            try:
                raw_compressed_bytes = bytes(msg.data)
                
                # 💡 print문 제거 (터미널 출력 방지)
                data = {
                    'topic': self.image_topic_name,
                    'sec': msg.header.stamp.sec,
                    'nanosec': msg.header.stamp.nanosec,
                    'data': raw_compressed_bytes,
                    'format': msg.format,
                    'is_compressed_raw': True
                }
                self.save_to_pickle(data)
            except KeyboardInterrupt:
                # Ctrl+C 인터럽트는 상위로 바로 던져서 즉시 죽도록 처리
                raise
            except Exception as e:
                self.get_logger().error(f"Top Compressed Color Image 저장 오류: {e}")

    def joint_states_callback(self, msg):
        topic_name = '/joint_states'
        if not self.should_save(topic_name): return

        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
        if msg_time > self.start_time:
            try:
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
            except KeyboardInterrupt:
                raise
            except Exception as e:
                pass

    def pose_stamped_callback(self, msg, topic_name):
        if not self.should_save(topic_name): return

        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
        if msg_time > self.start_time:
            try:
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
            except KeyboardInterrupt:
                raise
            except Exception as e:
                pass

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
        # 터미널에 에러 로그 패스하고 안전 종료
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass

if __name__ == '__main__':
    main()
