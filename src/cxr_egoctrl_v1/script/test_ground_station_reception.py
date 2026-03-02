#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import time
from geometry_msgs.msg import PoseArray, Pose, Point
from std_msgs.msg import Header

def test_cumulative_count():
    """测试累计数量功能"""
    
    # 等待ROS核心启动
    rospy.init_node('test_cumulative_count', anonymous=True)
    
    # 创建发布者
    waypoint_pub = rospy.Publisher('/planned_path', PoseArray, queue_size=1)
    
    # 等待发布者连接
    rospy.loginfo("等待发布者连接...")
    while waypoint_pub.get_num_connections() == 0:
        rospy.sleep(0.1)
    
    rospy.loginfo("✅ 发布者已连接，开始测试累计数量...")
    
    # 测试航点列表（按顺序访问预设的动物位置）
    test_waypoints = [
        # 1. 狼位置 - 应该累计 Wolf x2
        {
            "name": "狼位置测试",
            "world": [0.5, 1.5, 1.2],  # A6B2
            "expected_grid": "A6B2",
            "expected_animal": "Wolf x2",
            "expected_total": 2
        },
        # 2. 大象位置1 - 应该累计 Elephant x1
        {
            "name": "大象位置1测试",
            "world": [3.0, 3.0, 1.2],  # A3B7
            "expected_grid": "A3B7", 
            "expected_animal": "Elephant x1",
            "expected_total": 3
        },
        # 3. 老虎位置 - 应该累计 Tiger x1
        {
            "name": "老虎位置测试",
            "world": [2.0, 2.5, 1.2],  # A4B5
            "expected_grid": "A4B5",
            "expected_animal": "Tiger x1", 
            "expected_total": 4
        },
        # 4. 猴子位置 - 应该累计 Monkey x1
        {
            "name": "猴子位置测试",
            "world": [2.0, 3.0, 1.2],  # A3B5
            "expected_grid": "A3B5",
            "expected_animal": "Monkey x1",
            "expected_total": 5
        },
        # 5. 孔雀位置 - 应该累计 Peacock x1
        {
            "name": "孔雀位置测试", 
            "world": [0.5, 0.5, 1.2],  # A8B2
            "expected_grid": "A8B2",
            "expected_animal": "Peacock x1",
            "expected_total": 6
        },
        # 6. 随机选择位置 - 应该累计 Monkey或Peacock x1
        {
            "name": "随机选择测试",
            "world": [0.5, 3.5, 1.2],  # A2B2
            "expected_grid": "A2B2",
            "expected_animal": "Monkey或Peacock x1",
            "expected_total": 7
        },
        # 7. 大象位置2 - 应该累计 Elephant x1 (总数变为2)
        {
            "name": "大象位置2测试",
            "world": [2.5, 2.0, 1.2],  # A5B6
            "expected_grid": "A5B6",
            "expected_animal": "Elephant x1",
            "expected_total": 8
        }
    ]
    
    rospy.loginfo("�� 开始累计数量测试...")
    rospy.loginfo("�� 预期累计结果:")
    rospy.loginfo("  1. Wolf: 2")
    rospy.loginfo("  2. Elephant: 2") 
    rospy.loginfo("  3. Tiger: 1")
    rospy.loginfo("  4. Monkey: 1-2 (取决于随机选择)")
    rospy.loginfo("  5. Peacock: 1-2 (取决于随机选择)")
    rospy.loginfo("  6. 总计: 8-9")
    rospy.loginfo("")
    
    for i, test in enumerate(test_waypoints):
        rospy.loginfo(f"�� 测试 {i+1}/{len(test_waypoints)}: {test['name']}")
        rospy.loginfo(f"  目标网格: {test['expected_grid']}")
        rospy.loginfo(f"  期望动物: {test['expected_animal']}")
        rospy.loginfo(f"  期望累计总数: {test['expected_total']}")
        
        # 发布航点
        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = "map"
        
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = test['world']
        pose.orientation.w = 1.0
        pose_array.poses.append(pose)
        
        waypoint_pub.publish(pose_array)
        
        rospy.loginfo(f"  ✅ 已发布航点: {test['expected_grid']}")
        rospy.loginfo(f"  �� 世界坐标: ({test['world'][0]:.1f}, {test['world'][1]:.1f}, {test['world'][2]:.1f})")
        rospy.loginfo(f"  �� 动物检测发布器应该自动发布动物信息")
        rospy.loginfo(f"  �� 请检查累计数量是否正确")
        rospy.loginfo("")
        
        # 等待5秒让系统处理
        rospy.sleep(5.0)
    
    rospy.loginfo("�� 累计数量测试完成！")
    rospy.loginfo("")
    rospy.loginfo("�� 最终检查:")
    rospy.loginfo("1. 每种动物的累计数量是否正确？")
    rospy.loginfo("2. 总累计数量是否为8-9？")
    rospy.loginfo("3. 地面站显示的累计数据是否正确？")
    rospy.loginfo("")
    rospy.loginfo("�� 可以使用以下命令查看消息:")
    rospy.loginfo("rostopic echo /grid_animal_report")

if __name__ == '__main__':
    try:
        test_cumulative_count()
    except rospy.ROSInterruptException:
        rospy.loginfo("测试已停止")
