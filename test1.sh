
roslaunch point_lio mapping_mid360.launch & sleep 10;
roslaunch livox_ros_driver2 msg_MID360.launch & sleep 8;
rosrun tf static_transform_publisher 0 0 0.05 0 0 0 world camera_init 10  #& sleep 1;
#roslaunch mavros px4.launch & sleep 10;
#roslaunch camera_pose_node liopx4.launch  #& sleep 4;
#roslaunch target_ekf target_ekf.launch

wait;
