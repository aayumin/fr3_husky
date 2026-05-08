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
    t_file, c_file = f"tracker_{args.time}.csv", f"command_{args.time}.csv"
else:
    t_file, c_file = get_latest_file("tracker"), get_latest_file("command")

if not t_file or not os.path.exists(t_file):
    print("파일을 찾을 수 없습니다."); exit()

df_t = pd.read_csv(t_file)
df_c = pd.read_csv(c_file)


df_t['dt'] = df_t['time'].diff().fillna(0)
df_c['dt'] = df_c['time'].diff().fillna(0)

# 시작/끝 시간 설정
start_t = min(df_t['time'].min(), df_c['time'].min())
end_t = max(df_t['time'].max(), df_c['time'].max())

# --- 그래프 설정 (창 2개) ---
# Window 1: Tracker (2 Axes)
fig1, (ax1_pos, ax1_dt) = plt.subplots(2, 1, figsize=(8, 8))
fig1.canvas.manager.set_window_title('Tracker Data Playback')

# Window 2: Command (3 Axes)
fig2, (ax2_l, ax2_r, ax2_dt) = plt.subplots(3, 1, figsize=(10, 9))
fig2.canvas.manager.set_window_title('Command Data Playback')

# --- Line 객체 생성 (모든 관절/축) ---
# Tracker Pos (Left XYZ, Right XYZ)
ln_lx, = ax1_pos.plot([], [], label='L_x'); ln_ly, = ax1_pos.plot([], [], label='L_y'); ln_lz, = ax1_pos.plot([], [], label='L_z')
ln_rx, = ax1_pos.plot([], [], label='R_x'); ln_ry, = ax1_pos.plot([], [], label='R_y'); ln_rz, = ax1_pos.plot([], [], label='R_z')
ln_t_dt, = ax1_dt.plot([], [], 'k-', label='Tracker DT', linewidth=0.8)

# Command (L1~L7, R1~R7)
ln_l_joints = [ax2_l.plot([], [], label=f'L{i+1}')[0] for i in range(7)]
ln_r_joints = [ax2_r.plot([], [], label=f'R{i+1}')[0] for i in range(7)]
ln_c_dt, = ax2_dt.plot([], [], 'k-', label='Command DT', linewidth=0.8)

all_axes = [ax1_pos, ax1_dt, ax2_l, ax2_r, ax2_dt]


def update(current_t):
    xmin, xmax = current_t - 5.0, current_t

    # 데이터 필터링
    t_data = df_t[(df_t['time'] >= xmin) & (df_t['time'] <= xmax)]
    c_data = df_c[(df_c['time'] >= xmin) & (df_c['time'] <= xmax)]

    # 1. Tracker Pos 업데이트
    ln_lx.set_data(t_data['time'], t_data['lx']); ln_ly.set_data(t_data['time'], t_data['ly']); ln_lz.set_data(t_data['time'], t_data['lz'])
    ln_rx.set_data(t_data['time'], t_data['rx']); ln_ry.set_data(t_data['time'], t_data['ry']); ln_rz.set_data(t_data['time'], t_data['rz'])
    
    # 2. Tracker DT 업데이트 (기록된 dt가 있다면)
    if 'dt' in t_data.columns: ln_t_dt.set_data(t_data['time'], t_data['dt'])

    # 3. Command Joints 업데이트
    for i in range(7):
        ln_l_joints[i].set_data(c_data['time'], c_data[f'L{i+1}'])
        ln_r_joints[i].set_data(c_data['time'], c_data[f'R{i+1}'])
    
    # 4. Command DT 업데이트
    if 'dt' in c_data.columns: ln_c_dt.set_data(c_data['time'], c_data['dt'])

    # 공통 설정: 축 범위 및 자동 스케일
    for ax in all_axes:
        ax.set_xlim(xmin, xmax)
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)

    return []

# 애니메이션 실행 
frames = np.linspace(start_t, end_t, int((end_t - start_t) * 10))
ani1 = FuncAnimation(fig1, update, frames=frames, interval=20, blit=False, repeat=False)
ani2 = FuncAnimation(fig2, update, frames=frames, interval=20, blit=False, repeat=False)

# 범례/그리드 초기 설정
for ax in all_axes:
    ax.grid(True)
    ax.legend(loc='upper right', fontsize='small')

plt.tight_layout()
plt.show()
