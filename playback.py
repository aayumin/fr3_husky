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

parser = argparse.ArgumentParser()
parser.add_argument('--time', type=str, help='파일 이름의 시간 (예: 20260508_145808)')
args = parser.parse_args()

if args.time:
    t_file, c_file, x_file, target_x_l_file, target_x_r_file = f"tracker_{args.time}.csv", f"command_{args.time}.csv", f"x_m_{args.time}.csv", f"target_x_l_{args.time}.csv", f"target_x_r_{args.time}.csv"
else:
    t_file, c_file, x_file, target_x_l_file, target_x_r_file = get_latest_file("tracker"), get_latest_file("command"), get_latest_file("x_m"), get_latest_file("target_x_l"), get_latest_file("target_x_r")

if not t_file or not os.path.exists(t_file):
    print("파일을 찾을 수 없습니다."); exit()

df_t = pd.read_csv(t_file)
df_c = pd.read_csv(c_file)
df_x = pd.read_csv(x_file)
df_txl = pd.read_csv(target_x_l_file)
df_txr = pd.read_csv(target_x_r_file)


df_t['dt'] = df_t['time'].diff().fillna(0)
df_c['dt'] = df_c['time'].diff().fillna(0)
df_x['dt'] = df_x['time'].diff().fillna(0)
df_txl['dt'] = df_txl['time'].diff().fillna(0)
df_txr['dt'] = df_txr['time'].diff().fillna(0)

# 시작/끝 시간 설정
times_min = [df['time'].min() for df in [df_t, df_c, df_x] if not df.empty]
times_max = [df['time'].max() for df in [df_t, df_c, df_x] if not df.empty]
start_t = min(times_min) if times_min else 0.0
end_t = max(times_max) if times_max else 0.0


# --- 그래프 설정 ---
# Window 1: Tracker (2 Axes)
fig1, (ax1_pos, ax1_dt) = plt.subplots(2, 1, figsize=(8, 8))
fig1.canvas.manager.set_window_title('Tracker Data Playback')

# Window 2: target_X_pose (2 Axes)
fig2, (ax2_pos) = plt.subplots(1, 1, figsize=(8, 8))
fig2.canvas.manager.set_window_title('target_raw_x Data Playback')



# Window 3: current_X_pose (2 Axes)
fig3, (ax3_pos) = plt.subplots(1, 1, figsize=(8, 8))
fig3.canvas.manager.set_window_title('x_ Data Playback')


# Window 4: Command (3 Axes)
fig4, (ax4_l, ax4_r, ax4_dt) = plt.subplots(3, 1, figsize=(10, 9))
fig4.canvas.manager.set_window_title('Command Data Playback')

# --- Line 객체 생성 (모든 관절/축) ---
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
# Tracker Pos (Left XYZ, Right XYZ)
ln_lx, = ax1_pos.plot([], [], label='lx', color=colors[0]); ln_ly, = ax1_pos.plot([], [], label='ly', color=colors[1]); ln_lz, = ax1_pos.plot([], [], label='lz', color=colors[2])
ln_rx, = ax1_pos.plot([], [], label='rx', color=colors[3]); ln_ry, = ax1_pos.plot([], [], label='ry', color=colors[4]); ln_rz, = ax1_pos.plot([], [], label='rz', color=colors[5])
ln_t_dt, = ax1_dt.plot([], [], 'k-', label='Tracker DT', linewidth=0.8)

# target x pos (Left XYZ, Right XYZ)
ln_ltx, = ax2_pos.plot([], [], label='lx', color=colors[0]); ln_lty, = ax2_pos.plot([], [], label='ly', color=colors[1]); ln_ltz, = ax2_pos.plot([], [], label='lz', color=colors[2])
ln_rtx, = ax2_pos.plot([], [], label='rx', color=colors[3]); ln_rty, = ax2_pos.plot([], [], label='ry', color=colors[4]); ln_rtz, = ax2_pos.plot([], [], label='rz', color=colors[5])

# current x pos (Left XYZ, Right XYZ)
ln_lcx, = ax3_pos.plot([], [], label='lx', color=colors[0]); ln_lcy, = ax3_pos.plot([], [], label='ly', color=colors[1]); ln_lcz, = ax3_pos.plot([], [], label='lz', color=colors[2])
ln_rcx, = ax3_pos.plot([], [], label='rx', color=colors[3]); ln_rcy, = ax3_pos.plot([], [], label='ry', color=colors[4]); ln_rcz, = ax3_pos.plot([], [], label='rz', color=colors[5])

# Command (L1~L7, R1~R7)
ln_l_joints = [ax4_l.plot([], [], label=f'L{i+1}')[0] for i in range(7)]
ln_r_joints = [ax4_r.plot([], [], label=f'R{i+1}')[0] for i in range(7)]
ln_c_dt, = ax4_dt.plot([], [], 'k-', label='Command DT', linewidth=0.8)


all_axes = [ax1_pos, ax1_dt, ax2_pos, ax3_pos, ax4_l, ax4_r, ax4_dt]

initial_curxpos = None
initial_tracker_pos = None

def update(current_t):
    global initial_curxpos
    global initial_tracker_pos
    xmin, xmax = current_t - 5.0, current_t

    # 데이터 필터링
    t_data = df_t[(df_t['time'] >= xmin) & (df_t['time'] <= xmax)]
    c_data = df_c[(df_c['time'] >= xmin) & (df_c['time'] <= xmax)]
    x_data = df_x[(df_x['time'] >= xmin) & (df_x['time'] <= xmax)]
    txl_data = df_txl[(df_txl['time'] >= xmin) & (df_txl['time'] <= xmax)]
    txr_data = df_txr[(df_txr['time'] >= xmin) & (df_txr['time'] <= xmax)]
    


    # 1. Tracker Pos 업데이트
    if initial_curxpos is None and not x_data.empty:
        initial_curxpos = (x_data.loc[0, 'lx'], x_data.loc[0, 'ly'], x_data.loc[0, 'lz'], x_data.loc[0, 'rx'], x_data.loc[0, 'ry'], x_data.loc[0, 'rz'])
    if initial_tracker_pos is None and not t_data.empty:
        initial_tracker_pos = (t_data.loc[0, 'lx'], t_data.loc[0, 'ly'], t_data.loc[0, 'lz'], t_data.loc[0, 'rx'], t_data.loc[0, 'ry'], t_data.loc[0, 'rz'])
    if initial_curxpos is not None and not t_data.empty:
        t_data['lx'] = (t_data['lx'] - initial_tracker_pos[0]) + initial_curxpos[0]
        t_data['ly'] = (t_data['ly'] - initial_tracker_pos[1]) + initial_curxpos[1]
        t_data['lz'] = (t_data['lz'] - initial_tracker_pos[2]) + initial_curxpos[2]
        t_data['rx'] = (t_data['rx'] - initial_tracker_pos[3]) + initial_curxpos[3]
        t_data['ry'] = (t_data['ry'] - initial_tracker_pos[4]) + initial_curxpos[4]
        t_data['rz'] = (t_data['rz'] - initial_tracker_pos[5]) + initial_curxpos[5]
        ln_lx.set_data(t_data['time'], t_data['lx']); ln_ly.set_data(t_data['time'], t_data['ly']); ln_lz.set_data(t_data['time'], t_data['lz'])
        ln_rx.set_data(t_data['time'], t_data['rx']); ln_ry.set_data(t_data['time'], t_data['ry']); ln_rz.set_data(t_data['time'], t_data['rz'])

    if 'dt' in t_data.columns: ln_t_dt.set_data(t_data['time'], t_data['dt'])



    # 2. target Pos 업데이트
    ln_ltx.set_data(txl_data['time'], txl_data['lx']); ln_lty.set_data(txl_data['time'], txl_data['ly']); ln_ltz.set_data(txl_data['time'], txl_data['lz'])
    ln_rtx.set_data(txr_data['time'], txr_data['rx']); ln_rty.set_data(txr_data['time'], txr_data['ry']); ln_rtz.set_data(txr_data['time'], txr_data['rz'])

    # 3. Current Pos 업데이트
    ln_lcx.set_data(x_data['time'], x_data['lx']); ln_lcy.set_data(x_data['time'], x_data['ly']); ln_lcz.set_data(x_data['time'], x_data['lz'])
    ln_rcx.set_data(x_data['time'], x_data['rx']); ln_rcy.set_data(x_data['time'], x_data['ry']); ln_rcz.set_data(x_data['time'], x_data['rz'])


    # 4. Command Joints 업데이트
    for i in range(7):
        ln_l_joints[i].set_data(c_data['time'], c_data[f'L{i+1}'])
        ln_r_joints[i].set_data(c_data['time'], c_data[f'R{i+1}'])
    if 'dt' in c_data.columns: ln_c_dt.set_data(c_data['time'], c_data['dt'])

    # 공통 설정: 축 범위 및 자동 스케일
    for ax in all_axes:
        ax.set_xlim(xmin, xmax)
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)

    ax1_pos.set_ylim(-1.0, 1.0)
    ax2_pos.set_ylim(-1.0, 1.0)
    ax3_pos.set_ylim(-1.0, 1.0)

    return []


frames = np.linspace(start_t, end_t, int((end_t - start_t) * 10))
ani1 = FuncAnimation(fig1, update, frames=frames, interval=20, blit=False, repeat=False)
ani2 = FuncAnimation(fig2, update, frames=frames, interval=20, blit=False, repeat=False)
ani3 = FuncAnimation(fig3, update, frames=frames, interval=20, blit=False, repeat=False)
ani4 = FuncAnimation(fig4, update, frames=frames, interval=20, blit=False, repeat=False)

# 범례/그리드 초기 설정
for ax in all_axes:
    ax.grid(True)
    ax.legend(loc='upper right', fontsize='small')

plt.tight_layout()
plt.show()
