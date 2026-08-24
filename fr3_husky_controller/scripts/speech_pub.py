import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import os

class FilePublisher(Node):
    def __init__(self):
        super().__init__('file_publisher')
        
        # 1. 퍼블리셔 설정 (토픽명: 'speech', 큐 사이즈: 10)
        self.publisher_ = self.create_publisher(String, 'speech', 10)
        
        # 2. 절대 경로 파일 지정
        self.file_path = '/root/ssds_HL/recognized_speech.txt'
        
        # 3. 10Hz 타이머 설정 (10Hz = 0.1초 간격)
        timer_period = 0.1  
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('10Hz 파일 내용 퍼블리시 노드가 시작되었습니다.')

    def read_live_file(self):
        """실시간으로 파일 내용을 읽어오는 함수"""
        if not os.path.exists(self.file_path):
            return ""
            
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()  # 공백 및 줄바꿈 제거 후 반환
        except Exception as e:
            return ""

    def timer_callback(self):
        """0.1초마다 파일 확인 및 조건부 발행"""
        current_content = self.read_live_file()
        
        if not current_content:
            return

        msg = String()
        msg.data = current_content
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = FilePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
