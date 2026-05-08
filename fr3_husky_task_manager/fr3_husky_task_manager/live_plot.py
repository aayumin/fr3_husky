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

class TeleopPlotter(Node):
    def __init__(self):
        super().__init__("teleop_plotter")

        self.max_len = 500
        self.base_time = None

        ## /tracker_pose
        self.t_tracker = deque(maxlen=self.max_len)
        self.left_x = deque(maxlen=self.max_len)
        self.left_y = deque(maxlen=self.max_len)
        self.left_z = deque(maxlen=self.max_len)
        self.right_x = deque(maxlen=self.max_len)
        self.right_y = deque(maxlen=self.max_len)
        self.right_z = deque(maxlen=self.max_len)


        ## /command
        self.t_cmd = deque(maxlen=self.max_len)
        self.cmd_list = [deque(maxlen=self.max_len) for _ in range(14)]


        ## dt
        self.last_tracker_time = None
        self.tracker_dt = deque(maxlen=self.max_len)
        self.last_cmd_time = None
        self.cmd_dt = deque(maxlen=self.max_len)
        # self.tracker_dt.append(0.0)
        # self.cmd_dt.append(0.0)


        ## QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )


        self.create_subscription(
            PoseArray,
            "/tracker_pose",
            self.tracker_callback, 
            sensor_qos,
        )

        self.create_subscription(
            Float64MultiArray,
            "/debug/command_mani",
            self.command_callback,
            sensor_qos,
        )

        plt.ion()
        self.fig, self.axes = plt.subplots(2, 1, figsize=(6, 8))
        self.fig2, self.axes2 = plt.subplots(3, 1, figsize=(8, 8))

    def tracker_callback(self, msg):
        now = time.time()
        if self.base_time is None:
            self.base_time = now


        if self.last_tracker_time is None:
            self.last_tracker_time = now

            

        self.t_tracker.append(now - self.base_time)  # elapsed time
        self.tracker_dt.append(now - self.last_tracker_time)  # dt

        self.last_tracker_time = now


        left = msg.poses[0].position
        right = msg.poses[1].position


        self.left_x.append(left.x)
        self.left_y.append(left.y)
        self.left_z.append(left.z)

        self.right_x.append(right.x)
        self.right_y.append(right.y)
        self.right_z.append(right.z)

    def command_callback(self, msg):
        now = time.time()


        if self.base_time is None:
            self.base_time = now


        if self.last_cmd_time is None:
            self.last_cmd_time = now

            

        self.t_cmd.append(now - self.base_time)  # elapsed time
        self.cmd_dt.append(now - self.last_cmd_time)  # dt


        self.last_cmd_time = now

        
        for i in range(14):
            self.cmd_list[i].append(msg.data[i])


    def update_plot(self):
        if self.base_time is None: return
        current_elapsed = time.time() - self.base_time
        
        if current_elapsed < 5.0:
            xmin = 0.0
            xmax = 5.0
        else:
            xmin = current_elapsed - 5.0
            xmax = current_elapsed

            
        for ax in self.axes:
            ax.clear() 
            ax.set_xlim(xmin, xmax)
            ax.xaxis.set_major_locator(MultipleLocator(1.0))
            ax.grid(True, which="major", linestyle='-')
        
        for ax in self.axes2:
            ax.clear() 
            ax.set_xlim(xmin, xmax)
            ax.xaxis.set_major_locator(MultipleLocator(1.0))
            ax.grid(True, which="major", linestyle='-')

        # Subplot 0: Position
        self.axes[0].set_title("/tracker_pose hand position")
        self.axes[0].plot(self.t_tracker, self.left_x, label="left x")
        self.axes[0].plot(self.t_tracker, self.left_y, label="left y")
        self.axes[0].plot(self.t_tracker, self.left_z, label="left z")
        self.axes[0].plot(self.t_tracker, self.right_x, label="right x")
        self.axes[0].plot(self.t_tracker, self.right_y, label="right y")
        self.axes[0].plot(self.t_tracker, self.right_z, label="right z")
        self.axes[0].legend(loc="upper right")

        # Subplot 1: Tracker DT
        self.axes[1].set_title("/tracker_pose period dt")
        t_list = list(self.t_tracker)
        dt_list = list(self.tracker_dt)
        min_len = min(len(t_list), len(dt_list))
        self.axes[1].plot(t_list[-min_len:], dt_list[-min_len:], label="dt sec")

        # Subplot 2: Commands
        self.axes2[0].set_title("Left joint command")
        for i in range(7):
            self.axes2[0].plot(self.t_cmd, self.cmd_list[i], label = f"left joint_{i+1}")
        self.axes2[0].legend(loc="upper right")


        self.axes2[1].set_title("Right joint command")
        for i in range(7):
            self.axes2[1].plot(self.t_cmd, self.cmd_list[i+7], label = f"right joint_{i+1}")
        self.axes2[1].legend(loc="upper right")



        # Subplot 3: Command DT
        self.axes2[2].set_title("/command period dt")
        t_c_list = list(self.t_cmd)
        dt_c_list = list(self.cmd_dt)
        min_len_c = min(len(t_c_list), len(dt_c_list))
        self.axes2[2].plot(t_c_list[-min_len_c:], dt_c_list[-min_len_c:], label="dt sec")

        self.fig.tight_layout()
        self.fig2.tight_layout()
        plt.pause(0.001)


def main():
    rclpy.init()
    node = TeleopPlotter()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            node.update_plot()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
