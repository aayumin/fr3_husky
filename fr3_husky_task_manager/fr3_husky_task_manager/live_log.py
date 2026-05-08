import time
from collections import deque

import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float64MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
from matplotlib.ticker import MultipleLocator


class TeleopLogger(Node):
    def __init__(self):
        super().__init__("teleop_logger")
        self.max_len = 500
        self.base_time = None

        # 데이터 저장 (기존과 동일)
        self.t_tracker = deque(maxlen=self.max_len)
        self.left_x, self.left_y, self.left_z = [deque(maxlen=self.max_len) for _ in range(3)]
        self.right_x, self.right_y, self.right_z = [deque(maxlen=self.max_len) for _ in range(3)]
        self.t_cmd = deque(maxlen=self.max_len)
        self.cmd_list = [deque(maxlen=self.max_len) for _ in range(14)]

        # QoS 및 Sub (기존과 동일)
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(PoseArray, "/tracker_pose", self.tracker_callback, sensor_qos)
        self.create_subscription(Float64MultiArray, "/debug/command_mani", self.command_callback, sensor_qos)


        self.start_str = time.strftime('%Y%m%d_%H%M%S')
        self.save_targets = [
            (f"tracker_{self.start_str}.csv", self.t_tracker, 
            [self.left_x, self.left_y, self.left_z, self.right_x, self.right_y, self.right_z],
            ["time", "lx", "ly", "lz", "rx", "ry", "rz"]),
            
            (f"command_{self.start_str}.csv", self.t_cmd, 
            self.cmd_list, 
            ["time"] + [f"L{i+1}" for i in range(7)] + [f"R{i+1}" for i in range(7)])
        ]



    def tracker_callback(self, msg):
        now = time.time()
        if self.base_time is None: self.base_time = now
        self.t_tracker.append(now - self.base_time)
        
        l, r = msg.poses[0].position, msg.poses[1].position
        self.left_x.append(l.x); self.left_y.append(l.y); self.left_z.append(l.z)
        self.right_x.append(r.x); self.right_y.append(r.y); self.right_z.append(r.z)

    def command_callback(self, msg):
        now = time.time()
        if self.base_time is None: self.base_time = now

        self.t_cmd.append(now - self.base_time)
        for i in range(14): self.cmd_list[i].append(msg.data[i])

    def update(self):

        if self.base_time is None: return
        print("running...")
        plt.pause(0.001)


def main():
    rclpy.init()
    node = TeleopLogger()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            node.update()
    except KeyboardInterrupt:
            import csv
            node.get_logger().info('Saving data...')
            for filename, t_list, data_lists, headers in node.save_targets:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    # zip을 이용해 시간과 데이터를 한 줄씩 묶어 저장
                    for row in zip(t_list, *data_lists):
                        writer.writerow(row)
            node.get_logger().info('All data saved safely.')
        
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
