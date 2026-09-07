import os
import glob
import subprocess
import time
import sys
import signal
from datetime import datetime


processes = []

def run_command(command, bg=False):
    """ROS2 명령어를 실행하는 유틸리티 함수"""
    print(f"[EXEC] {command}")
    if bg:
        return subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
    else:
        return subprocess.run(command, shell=True)

def kill_all_background_processes():
    """켜져 있는 모든 백그라운드 프로세스 그룹을 안전하고 완벽하게 살해"""
    print("\n[INFO] 모든 백그라운드 로봇 프로세스를 정리합니다...")
    for p in processes:
        if p.poll() is None: # 아직 작동 중인 경우
            try:
                # 프로세스 그룹 전체에 SIGKILL 송신
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass

def get_latest_pkl(directory="pkl_data"):
    """pkl_data 폴더에서 가장 최근에 생성된 pkl 파일 경로를 반환"""
    if not os.path.exists(directory):
        return None
    list_of_files = glob.glob(os.path.join(directory, "*.pkl"))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def main():
    global processes
    
    # 디렉토리 미리 확인 및 생성
    if not os.path.exists("pkl_data"):
        os.makedirs("pkl_data")
        print("[INFO] 'pkl_data' 디렉토리를 확인/생성했습니다.")

    print("=== [Scenario #1] Block Stack Data Collection Script ===")

    try:
        # ----------------------------------------------------
        # 1. Realsense 카메라 실행 여부 확인
        # ----------------------------------------------------
        cam_reply = input(">> 1) 카메라(Realsense)를 실행하겠습니까? (y/n): ").strip().lower()
        if cam_reply == 'y':
            cam_proc = run_command("ros2 launch realsense2_camera rs_launch.py", bg=True)
            processes.append(cam_proc)
            time.sleep(3) # 카메라 초기화 대기
        
        # 기본 로봇 컨트롤러 실행
        print("\n[INFO] 로봇 컨트롤러를 실행합니다...")
        controller_cmd = "ros2 launch fr3_husky_controller fr3_action_controller.launch.py use_mujoco:=false load_mobile:=true robot_side:=dual"
        controller_proc = run_command(controller_cmd, bg=True)
        processes.append(controller_proc)
        time.sleep(5) # 컨트롤러 초기화 대기

        # ----------------------------------------------------
        # 2. 오브젝트 초기 위치 설정 및 EE 이동 (레퍼런스)
        # ----------------------------------------------------

        init_cmd = (
            f"ros2 run fr3_husky_task_manager move_to_joint "
            f"--max-velocity-scaling-factor 0.05 "
        )
        run_command(init_cmd)
        time.sleep(2)


        print("\n----------------------------------------------------")
        obj_reply = input(">> 2) 초기 첫번째 오브젝트 위치를 직접 지정하겠습니까? (y/n [default]): ").strip().lower()
        
        # TaskSpaceMoveClient 내 DEFAULT_RIGHT_POSE 기준 값 반영
        obj_x = 0.41
        obj_y = -0.25
        obj_z = 0.42
        obj_yaw = 0.0
        
        if obj_reply == 'y':
            try:
                obj_x = float(input(f"   - Object X 위치 (m) [현재:{obj_x}]: ") or obj_x)
                obj_y = float(input(f"   - Object Y 위치 (m) [현재:{obj_y}]: ") or obj_y)
                obj_yaw = float(input(f"   - Object Yaw 각도 (rad) [현재:{obj_yaw}]: ") or obj_yaw)
            except ValueError:
                print("   [WARN] 입력 값이 올바르지 않아 기본값으로 진행합니다.")
        
        # 사용자가 오브젝트를 안전하게 놓을 수 있도록 Z축을 10cm 올린 레퍼런스 위치 제공
        ref_z = obj_z + 0.05
        print("[INFO] 사용자가 오브젝트를 수동 배치할 수 있도록 EE를 레퍼런스 위치로 이동합니다...")
        
        # 💡 ROS2 Parameter 양식에 맞추어 명령 생성 (arm:=right 필수 지정)
        move_cmd_ref = (
            f"ros2 run fr3_husky_task_manager task_space_move  "
            f"--arm right "
            f"--duration 2.5 "
            f"--right-position {obj_x} {obj_y} {ref_z} "
            f"--right-rpy {3.141} {0.0} {obj_yaw}"
        )
        run_command(move_cmd_ref)
        
        input(">> [확인] 실제 첫번째 오브젝트 위치 조율 및 배치가 끝났다면 엔터[Enter]를 누르세요.")


        print("\n----------------------------------------------------")
        obj_reply = input(">> 2) 초기 두번째 오브젝트 위치를 직접 지정하겠습니까? (y/n [default]): ").strip().lower()
        
        # TaskSpaceMoveClient 내 DEFAULT_RIGHT_POSE 기준 값 반영
        obj_x = 0.45
        obj_y = -0.1
        obj_z = 0.42
        obj_yaw = 0.5
        
        if obj_reply == 'y':
            try:
                obj_x = float(input(f"   - Object X 위치 (m) [현재:{obj_x}]: ") or obj_x)
                obj_y = float(input(f"   - Object Y 위치 (m) [현재:{obj_y}]: ") or obj_y)
                obj_yaw = float(input(f"   - Object Yaw 각도 (rad) [현재:{obj_yaw}]: ") or obj_yaw)
            except ValueError:
                print("   [WARN] 입력 값이 올바르지 않아 기본값으로 진행합니다.")
        
        # 사용자가 오브젝트를 안전하게 놓을 수 있도록 Z축을 10cm 올린 레퍼런스 위치 제공
        ref_z = obj_z + 0.05
        print("[INFO] 사용자가 오브젝트를 수동 배치할 수 있도록 EE를 레퍼런스 위치로 이동합니다...")
        
        # 💡 ROS2 Parameter 양식에 맞추어 명령 생성 (arm:=right 필수 지정)
        move_cmd_ref = (
            f"ros2 run fr3_husky_task_manager task_space_move  "
            f"--arm right "
            f"--duration 2.5 "
            f"--right-position {obj_x} {obj_y} {ref_z} "
            f"--right-rpy {3.141} {0.0} {obj_yaw}"
        )
        run_command(move_cmd_ref)
        
        input(">> [확인] 실제 두번째 오브젝트 위치 조율 및 배치가 끝났다면 엔터[Enter]를 누르세요.")




        # ----------------------------------------------------
        # 3. Grid 설정 (Initial EE Position)
        # ----------------------------------------------------
        print("\n----------------------------------------------------")
        grid_reply = input(">> 3) Grid 오프셋 및 가로/세로 개수를 지정하겠습니까? (y/n [default]): ").strip().lower()
        
        # 기본값 설정: 4x4, 10cm 오프셋
        grid_rows, grid_cols = 4, 4
        grid_offset = 0.05
        
        if grid_reply == 'y':
            try:
                grid_rows = int(input("   - 가로 Grid 개수 (행) [기본:4]: ") or grid_rows)
                grid_cols = int(input("   - 세로 Grid 개수 (열) [기본:4]: ") or grid_cols)
                grid_offset = float(input("   - Grid 오프셋 간격 (m) [기본:0.10]: ") or grid_offset)
            except ValueError:
                print("   [WARN] 입력 값이 올바르지 않아 기본값으로 진행합니다.")

        total_grids = grid_rows * grid_cols
        print(f"[INFO] 총 {total_grids} 개의 Grid가 셋업되었습니다. (1 ~ {total_grids})")

        # ----------------------------------------------------
        # 4. Grid 인덱스 선택 및 초기 EE 이동
        # ----------------------------------------------------
        obj_x = 0.41
        obj_y = -0.25
        obj_z = 0.42
        obj_yaw = 0.0

        print("\n----------------------------------------------------")
        while True:
            try:
                grid_idx = int(input(f">> 4) 몇 번째 데이터 수집을 진행하시겠습니까? (1~{total_grids}): "))
                if 1 <= grid_idx <= total_grids:
                    break
                else:
                    print(f"   [WARN] 1부터 {total_grids} 사이의 숫자를 입력해주세요.")
            except ValueError:
                print("   [WARN] 숫자를 입력해주세요.")

        # Grid Index 기반 연산 (가운데 정렬 기준 격자 생성)
        row = (grid_idx - 1) // grid_cols
        col = (grid_idx - 1) % grid_cols
        
        target_ee_x = obj_x + (row - (grid_rows - 1)/2.0) * grid_offset
        target_ee_y = obj_y + (col - (grid_cols - 1)/2.0) * grid_offset
        target_ee_z = obj_z + 0.05 # 초기 10cm 오프셋을 준 EE 포지션
        
        print(f"[INFO] 선택한 Grid #{grid_idx} 초기 위치로 EE를 이동합니다.")
        print(f"       (Target X: {target_ee_x:.3f}, Y: {target_ee_y:.3f}, Z: {target_ee_z:.3f})")
        
        # 💡 ROS2 Parameter 양식에 맞추어 격자 타겟 위치 명령 생성
        move_cmd_grid = (
            f"ros2 run fr3_husky_task_manager task_space_move "
            f"--arm right "
            f"--duration 5.0 "
            f"--right-position {target_ee_x:.3f} {target_ee_y:.3f} {target_ee_z:.3f} "
            f"--right-rpy {3.141} {0.0} {0.0}"
        )
        run_command(move_cmd_grid)
        time.sleep(2)

        # ----------------------------------------------------
        # 5. 데이터 수집 시작 (Teleoperation & check2 Task)
        # ----------------------------------------------------
        print("\n----------------------------------------------------")
        input(">> 5) 준비 완료. 원격제어 및 데이터 수집을 시작하려면 [Enter]를 누르세요.")
        
        # teleop_proc = run_command("ros2 run fr3_husky_task_manager keyboard_move", bg=True)
        teleop_proc = run_command("ros2 run fr3_husky_task_manager apple_vision_pro --yaw-only --left-off", bg=True)
        processes.append(teleop_proc)
        
        data_collect_proc = run_command("ros2 run fr3_husky_task_manager check2", bg=True)
        processes.append(data_collect_proc)

        # ----------------------------------------------------
        # 6. 'q' 대기 루프
        # ----------------------------------------------------
        print("\n[RUNNING] ⏺️ 데이터 수집 중... 완료 후 'q' 입력 시 세이브 단계로 진입합니다.")
        while True:
            stop_signal = input().strip().lower()
            if stop_signal == 'q':
                print("[INFO] 수집 노드를 정지하고 정리 단계로 전환합니다.")
                # 원격 및 기록 전용 노드 먼저 제거하여 파일 해제
                os.killpg(os.getpgid(data_collect_proc.pid), signal.SIGKILL)
                os.killpg(os.getpgid(teleop_proc.pid), signal.SIGKILL)
                time.sleep(1.0)
                break

        # ----------------------------------------------------
        # 7. 데이터 저장(Save) 또는 폐기(Discard) 선택 처리
        # ----------------------------------------------------
        print("\n----------------------------------------------------")
        decision = input(">> 6) 이번에 수집된 시퀀스를 저장하시겠습니까? (save / discard): ").strip().lower()
        latest_file = get_latest_pkl("pkl_data")

        if decision == 'save':
            if latest_file and os.path.exists(latest_file):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_filename = f"pkl_data/stack_grid{grid_idx}_{timestamp}.pkl"
                os.rename(latest_file, final_filename)
                print(f"[SUCCESS] ✨ 데이터가 저장되었습니다 -> {final_filename}")
            else:
                print("[ERROR] 임시 생성된 pkl 데이터를 찾지 못했습니다.")
        else:
            if latest_file and os.path.exists(latest_file):
                os.remove(latest_file)
                print(f"[INFO] 🗑️ 수집 데이터를 성공적으로 폐기했습니다.")

    except KeyboardInterrupt:
        # 어떤 스테이지에서든 사용자가 Ctrl+C를 누르면 이쪽으로 즉시 튕겨 나옵니다.
        print("\n\n[USER_QUIT] ⚠️ 사용자에 의해 데이터 수집 프로세스가 전면 중단되었습니다.")
    finally:
        # 정상 종료든, Ctrl+C 강제 종료든 무조건 실행되어 백그라운드 노드 전체 살해
        kill_all_background_processes()
        print("[INFO] 스크립트 실행을 종료합니다.")

if __name__ == "__main__":
    main()