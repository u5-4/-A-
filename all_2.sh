#!/bin/bash

# === 环境初始化 ===
source /opt/ros/noetic/setup.bash
source /home/nvidia/ws_livox/devel/setup.bash
source /home/nvidia/ws_pointlio2/devel/setup.bash
source /home/nvidia/ws_ctrlpx4/devel/setup.bash
source /home/nvidia/ros_ws/devel/setup.bash

# === ROS通信层 ===
roslaunch rosbridge_server rosbridge_websocket.launch & sleep 3

# === 感知系统 ===
roslaunch point_lio mapping_mid360.launch & sleep 12
roslaunch livox_ros_driver2 msg_MID360.launch & sleep 8
rosrun tf static_transform_publisher 0 0 0 0 0 0 world map 100 & sleep 1
roslaunch astra_camera astra_pro.launch & sleep 15

# === 飞控系统 ===
roslaunch mavros px4.launch & sleep 10
roslaunch camera_pose_node liopx4.launch & sleep 4

# === 核心算法模块 ===
python3 /home/nvidia/ws_ctrlpx4/src/cxr_egoctrl_v1/script/animal_detection_publisher.py & sleep 3
python3 /home/nvidia/ws_ctrlpx4/src/cxr_egoctrl_v1/script/coordinate_converter.py & sleep 2
python3 /home/nvidia/ws_ctrlpx4/src/cxr_egoctrl_v1/script/path_planner_enhanced.py & sleep 2
python3 /home/nvidia/ws_ctrlpx4/src/cxr_egoctrl_v1/script/auto_path_publisher.py & sleep 3
# 新增网格报告发布器
python3 /home/nvidia/ws_ctrlpx4/src/cxr_egoctrl_v1/script/grid_report_publisher.py & sleep 2  # <-- 新增行

# === 规划与控制系统 ===
roslaunch ego_planner single_drone_exp.launch & sleep 10
roslaunch ego_planner rviz.launch & sleep 5
roslaunch target_ekf target_ekf.launch &

# === 飞控核心节点（新终端）===
gnome-terminal --tab -- bash -c "roslaunch px4ctrl run_ctrl.launch; exec bash" & sleep 5

# 等待所有后台任务
wait
