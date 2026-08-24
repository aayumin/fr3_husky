# FR3 Husky ROS 2 Workspace

`fr3_husky`는 Franka Research 3(FR3) 로봇 팔과 Husky 이동 로봇을 제어하기 위한 ROS 2 패키지입니다.
이 저장소는 기본 로봇 제어와 ROS 2 Action Goal 기반 작업 실행을 제공합니다.

원격조작 기능은 아래 연동 저장소를 함께 사용합니다.

* **Apple Vision Pro 손 추적 기반 원격조작**
  [AppleVisionPro_HandTracking](https://github.com/aayumin/AppleVisionPro_HandTracking.git)

* **음성·텍스트 명령 기반 원격조작**
  [ssds_HL](https://github.com/smtamh/ssds_HL.git)

각 원격조작 기능의 설정과 실행 방법은 해당 저장소의 README를 참고하세요.

## 요구 사항

* Ubuntu 22.04
* ROS 2 Humble
* [libfranka](https://frankarobotics.github.io/docs/libfranka/docs/installation.html)

> `libfranka` 설치 시 `franka_ros2`와 `franka_description`은 함께 설치하지 마세요. 아래에서 안내하는 소스 의존성을 사용합니다.

## 설치

### 1. ROS 2 작업 공간 준비

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

### 2. 필수 저장소 내려받기

```bash
git clone https://github.com/aayumin/fr3_husky.git
git clone https://github.com/JunHeonYoon/franka_ros2.git
git clone https://github.com/JunHeonYoon/franka_description.git
git clone https://github.com/JunHeonYoon/husky.git
git clone https://github.com/JunHeonYoon/dyros_robot_controller.git
```

MuJoCo 시뮬레이션을 사용할 경우에는 아래 저장소도 내려받으세요.

```bash
git clone https://github.com/JunHeonYoon/mujoco_ros_hardware.git
```

### 3. 의존성 설치 및 빌드

```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-up-to fr3_husky_task_manager
```

### 4. 작업 공간 적용

새 터미널을 열 때마다 아래 명령을 실행하세요.

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

## 로봇 시각화

FR3 로봇 팔만 시각화하려면 다음 명령을 실행합니다.

```bash
ros2 launch fr3_husky_description visualize_fr3.launch.py \
  robot_side:=left load_gripper:=true load_mobile:=false
```

FR3와 Husky를 함께 시각화하려면 다음 명령을 실행합니다.

```bash
ros2 launch fr3_husky_description visualize_fr3_husky.launch.py \
  robot_side:=left load_gripper:=true
```

`robot_side`에는 `left`, `right`, `dual`을 지정할 수 있습니다.

## 로봇 제어 실행

### FR3 로봇 팔 제어

```bash
ros2 launch fr3_husky_controller fr3_action_controller.launch.py \
  robot_side:=left load_gripper:=true load_mobile:=false \
  use_fake_hardware:=true
```

### FR3와 Husky 통합 제어

```bash
ros2 launch fr3_husky_controller fr3_husky_action_controller.launch.py \
  robot_side:=left load_gripper:=true \
  use_fake_hardware:=true joy_dev:=/dev/input/js0
```

시뮬레이션 또는 가상 하드웨어를 사용할 때는 `use_fake_hardware:=true`를 사용하세요.
실제 로봇을 제어할 때는 로봇 연결 및 안전 상태를 확인한 후 `use_fake_hardware:=false`로 실행하세요.

## Apple Vision Pro 손 추적 원격조작

Apple Vision Pro를 이용한 손 추적 원격조작은 별도 저장소에서 설정합니다.

1. 이 저장소에서 로봇 제어 launch를 실행합니다.
2. 아래 명령으로 작업 관리 노드를 실행합니다.

```bash
ros2 run fr3_husky_task_manager apple_vision_pro
```

3. [AppleVisionPro_HandTracking README](https://github.com/aayumin/AppleVisionPro_HandTracking)로 이동하여 안내에 따라 Apple Vision Pro 손 추적 프로그램을 설정하고 실행합니다.

## 음성·텍스트 명령 원격조작

음성 또는 텍스트 명령을 통해 로봇 작업을 수행하려면 다음 순서로 진행하세요.

1. 이 저장소에서 로봇 제어 launch를 실행합니다.
2. [ssds_HL README](https://github.com/smtamh/ssds_HL)로 이동합니다.
3. README의 안내에 따라 MCP 서버와 명령 인터페이스를 설정합니다.
4. 음성 또는 텍스트 명령을 입력하면 AI가 명령을 해석하여 적절한 ROS 2 Action Goal을 로봇에 전달합니다.

## 안전 안내

* 실제 로봇을 실행하기 전에 비상 정지 장치, 작업 공간, 네트워크 연결 상태를 확인하세요.
* 로봇 주변에 사람이나 장애물이 없는지 확인한 뒤 제어를 시작하세요.
* 처음 사용할 때는 반드시 시뮬레이션 또는 가상 하드웨어 환경에서 동작을 확인하는 것을 권장합니다.
