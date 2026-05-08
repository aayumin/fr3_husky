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
        self.last_plot_time = time.time() # 그리기 주기 조절용

        # 데이터 저장 (기존과 동일)
        self.t_tracker = deque(maxlen=self.max_len)
        self.left_x, self.left_y, self.left_z = [deque(maxlen=self.max_len) for _ in range(3)]
        self.right_x, self.right_y, self.right_z = [deque(maxlen=self.max_len) for _ in range(3)]
        self.t_cmd = deque(maxlen=self.max_len)
        self.cmd_list = [deque(maxlen=self.max_len) for _ in range(14)]
        self.tracker_dt = deque(maxlen=self.max_len)
        self.cmd_dt = deque(maxlen=self.max_len)
        self.last_tracker_time = None
        self.last_cmd_time = None

        # QoS 및 Sub (기존과 동일)
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(PoseArray, "/tracker_pose", self.tracker_callback, sensor_qos)
        self.create_subscription(Float64MultiArray, "/debug/command_mani", self.command_callback, sensor_qos)

        # Matplotlib 설정
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 1, figsize=(6, 8))
        self.fig2, self.axes2 = plt.subplots(3, 1, figsize=(8, 8))

        # --- 선(Line) 객체들 미리 생성 ---
        # Figure 1
        self.ln_lx, = self.axes[0].plot([], [], label="left x")
        self.ln_ly, = self.axes[0].plot([], [], label="left y")
        self.ln_lz, = self.axes[0].plot([], [], label="left z")
        self.ln_rx, = self.axes[0].plot([], [], label="right x")
        self.ln_ry, = self.axes[0].plot([], [], label="right y")
        self.ln_rz, = self.axes[0].plot([], [], label="right z")
        self.ln_t_dt, = self.axes[1].plot([], [], label="dt sec")

        # Figure 2
        self.ln_lcmds = [self.axes2[0].plot([], [], label=f"L{i+1}")[0] for i in range(7)]
        self.ln_rcmds = [self.axes2[1].plot([], [], label=f"R{i+1}")[0] for i in range(7)]
        self.ln_c_dt, = self.axes2[2].plot([], [], label="dt sec")

        # 초기 설정 (제목, 그리드 등 한 번만 실행)
        for i, ax in enumerate(list(self.axes) + list(self.axes2)):
            ax.grid(True)
            ax.xaxis.set_major_locator(MultipleLocator(1.0))
            if i == 0: ax.set_title("Tracker Position")
            if i == 1: ax.set_title("Tracker DT")


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
        if self.last_tracker_time is not None:
            self.tracker_dt.append(now - self.last_tracker_time)
        self.last_tracker_time = now
        self.t_tracker.append(now - self.base_time)
        
        l, r = msg.poses[0].position, msg.poses[1].position
        self.left_x.append(l.x); self.left_y.append(l.y); self.left_z.append(l.z)
        self.right_x.append(r.x); self.right_y.append(r.y); self.right_z.append(r.z)

    def command_callback(self, msg):
        now = time.time()
        if self.base_time is None: self.base_time = now
        if self.last_cmd_time is not None:
            self.cmd_dt.append(now - self.last_cmd_time)


        self.last_cmd_time = now
        self.t_cmd.append(now - self.base_time)
        for i in range(14): self.cmd_list[i].append(msg.data[i])

    def update_plot(self):

        if self.base_time is None: return
        
        # [성능 핵심] 30Hz 정도로 그리기 제한 (약 0.033초)
        now = time.time()
        if now - self.last_plot_time < 0.03:
            return
        self.last_plot_time = now

        current_elapsed = now - self.base_time
        xmin = max(0.0, current_elapsed - 5.0)
        xmax = max(5.0, current_elapsed)

        # 1. 데이터 업데이트 (set_data는 매우 빠름)
        self.ln_lx.set_data(self.t_tracker, self.left_x)
        self.ln_ly.set_data(self.t_tracker, self.left_y)
        self.ln_lz.set_data(self.t_tracker, self.left_z)
        self.ln_rx.set_data(self.t_tracker, self.right_x)
        self.ln_ry.set_data(self.t_tracker, self.right_y)
        self.ln_rz.set_data(self.t_tracker, self.right_z)
        
        n_tdt = min(len(self.t_tracker), len(self.tracker_dt))
        self.ln_t_dt.set_data(list(self.t_tracker)[-n_tdt:], list(self.tracker_dt)[-n_tdt:])

        for i in range(7):
            self.ln_lcmds[i].set_data(self.t_cmd, self.cmd_list[i])
            self.ln_rcmds[i].set_data(self.t_cmd, self.cmd_list[i+7])
        
        n_cdt = min(len(self.t_cmd), len(self.cmd_dt))
        self.ln_c_dt.set_data(list(self.t_cmd)[-n_cdt:], list(self.cmd_dt)[-n_cdt:])

        # 2. 축 범위 업데이트
        for ax in list(self.axes) + list(self.axes2):
            ax.set_xlim(xmin, xmax)
            # y축 자동 조절 (필요시)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

        # 3. 그리기 실행
        self.fig.canvas.flush_events()
        self.fig2.canvas.flush_events()
        plt.pause(0.001)


def main():
    rclpy.init()
    node = TeleopPlotter()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.005)
            node.update_plot()
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
