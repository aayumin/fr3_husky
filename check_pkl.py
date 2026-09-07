import os
import glob
import pickle
import math
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import io

def get_latest_pkl(directory="pkl_data"):
    """pkl_data 폴더에서 가장 최근에 생성된 pkl 파일 경로를 반환"""
    list_of_files = glob.glob(os.path.join(directory, "*.pkl"))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def quaternion_to_rpy(x, y, z, w):
    """쿼터니언을 Roll, Pitch, Yaw (라디안)로 변환"""
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z

def main():
    # 1. Argument Parser 설정
    parser = argparse.ArgumentParser(description="수집 데이터 (.pkl) 검증 및 플로팅 스크립트")
    parser.add_argument(
        "file_path", 
        nargs="?", 
        default=None, 
        help="검증할 pkl 파일 경로 (생략 시 가장 최근 수집된 파일 자동 선택)"
    )
    args = parser.parse_args()

    print("=== 수집 데이터 (.pkl) 검증 및 플로팅 스크립트 ===")
    
    # 2. 파일 경로 결정
    if args.file_path:
        target_file = args.file_path
        if not os.path.exists(target_file):
            print(f"[ERROR] 지정하신 파일이 존재하지 않습니다: {target_file}")
            return
    else:
        target_file = get_latest_pkl("pkl_data")
        if not target_file:
            print("[ERROR] pkl_data 폴더에 pkl 파일이 존재하지 않습니다.")
            return
    
    print(f"[INFO] 대상 파일 로드 중: {target_file}")
    
    # 데이터 파싱용 컨테이너
    images = []
    ee_time = []
    ee_x, ee_y, ee_z = [], [], []
    ee_roll, ee_pitch, ee_yaw = [], [], []
    
    # 3. Pickle 파일 순차 로드 (Streaming read)
    count = 0
    with open(target_file, "rb") as f:
        while True:
            try:
                data = pickle.load(f)
                count += 1
                
                topic = data.get('topic', '')
                timestamp = data.get('sec', 0) + data.get('nanosec', 0) * 1e-9
                
                # 이미지 토픽 분리
                if 'compressed' in topic:
                    images.append(data['data']) # JPEG/PNG 원본 바이트
                    
                # 우측 팔 현재 Pose 데이터 분리 (/debug/cur_pose_right)
                elif topic == '/debug/cur_pose_right':
                    pos = data['position']
                    ori = data['orientation']
                    
                    ee_time.append(timestamp)
                    ee_x.append(pos['x'])
                    ee_y.append(pos['y'])
                    ee_z.append(pos['z'])
                    
                    r, p, y = quaternion_to_rpy(ori['x'], ori['y'], ori['z'], ori['w'])
                    ee_roll.append(r)
                    ee_pitch.append(p)
                    ee_yaw.append(y)
                    
            except EOFError:
                break # 파일 끝 도달
                
    print(f"[INFO] 파싱 완료! 총 {count} 개의 데이터 프레임 감지.")
    print(f"       - 이미지 수집 개수: {len(images)} 프레임")
    print(f"       - EE Pose 수집 개수: {len(ee_time)} 개")

    if not ee_time:
        print("[WARN] EE Pose 데이터가 생성되지 않았습니다. 토픽명을 확인하세요.")
        return

    # 상대 시간(초)으로 변환
    start_t = ee_time[0]
    ee_time = [t - start_t for t in ee_time]

    # 4. 데이터 시각화 (Matplotlib Subplots)
    print("[INFO] 그래프(plt.plot)를 렌더링합니다...")
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 💡 4-1. Position (X, Y, Z) 플롯 -> axs[0]으로 인덱싱 지정
    axs[0].plot(ee_time, ee_x, 'r-', label='X Position')
    axs[0].plot(ee_time, ee_y, 'g-', label='Y Position')
    axs[0].plot(ee_time, ee_z, 'b-', label='Z Position')
    axs[0].set_ylabel('Position (meters)')
    axs[0].set_title(f"End-Effector Trajectory\n({os.path.basename(target_file)})")
    axs[0].legend(loc='upper right')
    axs[0].grid(True)
    
    # 💡 4-2. Orientation (RPY) 플롯 -> axs[1]로 인덱싱 지정
    axs[1].plot(ee_time, ee_roll, 'r--', label='Roll')
    axs[1].plot(ee_time, ee_pitch, 'g--', label='Pitch')
    axs[1].plot(ee_time, ee_yaw, 'b--', label='Yaw')
    axs[1].set_ylabel('Orientation (radians)')
    axs[1].set_xlabel('Relative Time (seconds)')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)
    
    plt.tight_layout()
    
    # 💡 파일 이름 튜플 파싱 버그 수정 (os.path.splitext 결과물 반영)
    base_name = os.path.splitext(os.path.basename(target_file))[0]
    save_img_path = f"pkl_data/{base_name}_trajectory.png"
    plt.savefig(save_img_path) 
    print(f"[SUCCESS] 그래프 이미지가 '{save_img_path}'에 저장되었습니다.")
    
    # 5. Realsense 비디오 이미지 재생
    if len(images) > 0:
        print("\n[INFO] 수집된 Realsense 비디오 화면을 재생합니다... (창을 닫으려면 ESC 또는 닫기 버튼)")
        plt.figure("Realsense Video Playback", figsize=(5, 5))
        
        # 첫 번째 프레임으로 가시화 창 초기화
        img_buf = io.BytesIO(images[0])
        img = Image.open(img_buf)
        im_display = plt.imshow(img)
        plt.axis('off')
        
        for idx, img_bytes in enumerate(images):
            if not plt.fignum_exists("Realsense Video Playback"):
                break # 창이 닫히면 재생 중지
                
            img_buf = io.BytesIO(img_bytes)
            img = Image.open(img_buf)
            
            im_display.set_data(img)
            plt.title(f"Playback: Frame {idx+1}/{len(images)}")
            plt.pause(1.0 / 30.0) # 30 FPS 속도로 재생 시도
    else:
        print("[WARN] 재생할 이미지 데이터가 파일 내에 없습니다.")

    plt.show()

if __name__ == "__main__":
    main()
