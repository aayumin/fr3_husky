import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import argparse
import glob
import os

# 1. 파일 선택 로직
def get_latest_file(prefix):
    files = glob.glob(f"{prefix}_*.csv")
    return max(files, key=os.path.getmtime) if files else None


def get_files_at_time(tt = None):
    if tt is None:
        last_fname = get_latest_file("tracker")
        tt = "_".join(last_fname.split(".")[0].split("_")[-2:])
        print(int(tt.split("_")[0]), int(tt.split("_")[1]))


    prefix_list = ["tracker",
                  "command",
                  "x_m",
                  "target_x_l",
                  "target_x_r",
                  "smooth_target_x_l",
                  "smooth_target_x_r",]
        
    fname_list = [None for _ in range(len(prefix_list))]
    for i, prefix in enumerate(prefix_list):
        fname_list[i] = f"{prefix}_{tt}.csv"
        
    fname_list = [fname_list[0], fname_list[1], fname_list[2], (fname_list[3], fname_list[4]), (fname_list[5], fname_list[6])]
    return fname_list


    

    

parser = argparse.ArgumentParser()
parser.add_argument('--time', type=str, help='파일 이름의 시간 (예: 20260508_145808)')
args = parser.parse_args()


fname_list = get_files_at_time(args.time)


if not fname_list[0] or not os.path.exists(fname_list[0]):
    print("파일을 찾을 수 없습니다."); exit()

########################

show_tracker = True
show_command = False
show_cur_x = True
show_raw_target = True
show_smooth_target = True
show_flags = [show_tracker, show_command, show_cur_x, show_raw_target, show_smooth_target,  ]

########################



df_list = []
for is_flag, fname in  list(zip(show_flags, fname_list)):
    if not is_flag: continue

    if isinstance(fname, tuple):
        ddf_list = []
        for ff in fname:
            df = pd.read_csv(ff)
            ddf_list.append(df)
        df_list.append(ddf_list)
    else:
        df = pd.read_csv(fname)
        df_list.append(df)




for df in df_list:
    if isinstance(df, list):
        for i, dff in enumerate(df):
            df[i]['dt'] = dff['time'].diff().fillna(0)
    else: df['dt'] = df['time'].diff().fillna(0)


# 시작/끝 시간 설정
times_min = [df['time'].min() for df in df_list if not isinstance(df, list) and not df.empty]
times_max = [df['time'].max() for df in df_list if not isinstance(df, list) and not df.empty]
for df in df_list:
    if isinstance(df, list):
        aaa = [dff['time'].min() for dff in df]
        times_min.extend(aaa)
        bbb = [dff['time'].max() for dff in df]
        times_max.extend(bbb)
start_t = min(times_min) if times_min else 0.0
end_t = max(times_max) if times_max else 0.0


# --- 그래프 설정 ---
window_list = []

# Window: Tracker (2 Axes)
if show_tracker:
    fig, (ax_pos, ax_dt) = plt.subplots(2, 1, figsize=(8, 16))
    fig.canvas.manager.set_window_title('Tracker Data Playback')
    window_list.append((fig, [ax_pos, ax_dt]))

# Window: 
if show_raw_target:
    fig, (ax_pos) = plt.subplots(1, 1, figsize=(8, 6))
    fig.canvas.manager.set_window_title('target_raw_x Data Playback')
    window_list.append((fig, [ax_pos]))

# Window: 
if show_smooth_target:
    fig, (ax_pos) = plt.subplots(1, 1, figsize=(8, 6))
    fig.canvas.manager.set_window_title('target_smooth_x Data Playback')
    window_list.append((fig, [ax_pos]))


# Window: current_X_pose (2 Axes)
if show_cur_x:
    fig, (ax_pos) = plt.subplots(1, 1, figsize=(8, 6))
    fig.canvas.manager.set_window_title('current x pos')
    window_list.append((fig, [ax_pos]))


# Window:
if show_command:
    fig, (ax_l, ax_r, ax_dt) = plt.subplots(3, 1, figsize=(10, 9))
    fig.canvas.manager.set_window_title('Command Data Playback')
    window_list.append((fig, [ax_l, ax_r, ax_dt]))



# --- Line 객체 생성 (모든 관절/축) ---
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']


ln_savings = []

for fig, axes in window_list:
    ln_lx, = axes[0].plot([], [], label='lx', color=colors[0]); ln_ly, = axes[0].plot([], [], label='ly', color=colors[1]); ln_lz, = axes[0].plot([], [], label='lz', color=colors[2])
    ln_rx, = axes[0].plot([], [], label='rx', color=colors[3]); ln_ry, = axes[0].plot([], [], label='ry', color=colors[4]); ln_rz, = axes[0].plot([], [], label='rz', color=colors[5])
    ln_t_dt = None
    if len(axes) > 1: 
        ln_t_dt, = axes[-1].plot([], [], 'k-', label='DT', linewidth=0.8)

    # axes[0].set_ylim(-1.0, 1.0)
    ln_savings.append([ln_lx, ln_ly, ln_lz, ln_rx, ln_ry, ln_rz, ln_t_dt])


# # Command (L1~L7, R1~R7)
# ln_l_joints = [ax4_l.plot([], [], label=f'L{i+1}')[0] for i in range(7)]
# ln_r_joints = [ax4_r.plot([], [], label=f'R{i+1}')[0] for i in range(7)]
# ln_c_dt, = ax4_dt.plot([], [], 'k-', label='Command DT', linewidth=0.8)


all_axes = []
for fig, axes in window_list: all_axes.extend(axes)


initial_curxpos = None
initial_tracker_pos = None

def update(current_t):
    global initial_curxpos
    global initial_tracker_pos
    xmin, xmax = current_t - 15.0, current_t

    # 데이터 필터링
    data_filter_list = []
    for df in df_list:
        if isinstance(df, list):
            data = []
            for dff in df:
                tempdata = dff[(dff['time'] >= xmin) & (dff['time'] <= xmax)]
                data.append(tempdata)

            data_filter_list.append(data)

        else:
            data = df[(df['time'] >= xmin) & (df['time'] <= xmax)]
            data_filter_list.append(data)


    # ax 업데이트
    for i, (data) in enumerate(data_filter_list):
        ln_lx, ln_ly, ln_lz, ln_rx, ln_ry, ln_rz, ln_t_dt = ln_savings[i]
        if isinstance(data, list):
            ldata = data[0]
            rdata = data[1]

            ln_lx.set_data(ldata['time'], ldata['lx']); ln_ly.set_data(ldata['time'], ldata['ly']); ln_lz.set_data(ldata['time'], ldata['lz'])
            ln_rx.set_data(rdata['time'], rdata['rx']); ln_ry.set_data(rdata['time'], rdata['ry']); ln_rz.set_data(rdata['time'], rdata['rz'])
            
        else:
            ln_lx.set_data(data['time'], data['lx']); ln_ly.set_data(data['time'], data['ly']); ln_lz.set_data(data['time'], data['lz'])
            ln_rx.set_data(data['time'], data['rx']); ln_ry.set_data(data['time'], data['ry']); ln_rz.set_data(data['time'], data['rz'])
        if ln_t_dt is not None: ln_t_dt.set_data(data['time'], data['dt'])

    # # 4. Command Joints 업데이트
    # for i in range(7):
    #     ln_l_joints[i].set_data(c_data['time'], c_data[f'L{i+1}'])
    #     ln_r_joints[i].set_data(c_data['time'], c_data[f'R{i+1}'])
    # if 'dt' in c_data.columns: ln_c_dt.set_data(c_data['time'], c_data['dt'])

    # 공통 설정: 축 범위 및 자동 스케일
    for ax in all_axes:
        ax.set_xlim(xmin, xmax)
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)

    return []





frames = np.linspace(start_t, end_t, int((end_t - start_t) * 2))
print(start_t, end_t)
# raise()

animations = []

for fig, axes in window_list:
    # 각 fig마다 고유한 ani 객체를 생성하고 리스트에 추가
    ani = FuncAnimation(fig, update, frames=frames, interval=20, blit=False, repeat=False)
    animations.append(ani)

# 범례/그리드 초기 설정
for ax in all_axes:
    ax.grid(True)
    ax.legend(loc='upper right', fontsize='small')

plt.tight_layout()
plt.show()
