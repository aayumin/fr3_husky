pkill -9 -f ros2; 
pkill -9 -f rclpy;
pkill -9 -f realsense2_camera
pkill -9 -f rs_launch

ros2 daemon stop
ros2 daemon start

python3 collect_all_data_stack.py
