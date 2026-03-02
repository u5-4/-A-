#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新的网格报告格式
"""
import rospy
import json
from std_msgs.msg import String

class NewGridFormatTester:
    def __init__(self):
        rospy.init_node('new_grid_format_tester', anonymous=True)
        
        # 订阅器
        self.grid_report_sub = rospy.Subscriber('/grid_animal_report', String, self.grid_report_callback)
        
        # 状态变量
        self.report_count = 0
        
        rospy.loginfo("新网格格式测试器已启动")
        rospy.loginfo("监控话题: /grid_animal_report")
        
    def grid_report_callback(self, msg):
        """网格报告回调"""
        self.report_count += 1
        
        try:
            grid_data = json.loads(msg.data)
            rospy.loginfo(f"�� 收到新格式网格报告 #{self.report_count}:")
            
            # 检查格式是否符合预期
            expected_fields = [
                "grid_position", "animals", "total_animals", 
                "grid_total", "mission_total", "drone_position", "report_type"
            ]
            
            missing_fields = []
            for field in expected_fields:
                if field not in grid_data:
                    missing_fields.append(field)
            
            if missing_fields:
                rospy.logwarn(f"❌ 缺少字段: {missing_fields}")
            else:
                rospy.loginfo("✅ 格式完整")
            
            # 显示详细信息
            rospy.loginfo(f"  网格位置: {grid_data.get('grid_position', 'N/A')}")
            rospy.loginfo(f"  网格动物: {grid_data.get('animals', {})}")
            rospy.loginfo(f"  网格总数: {grid_data.get('grid_total', 0)}")
            rospy.loginfo(f"  任务总数: {grid_data.get('mission_total', 0)}")
            rospy.loginfo(f"  无人机位置: {grid_data.get('drone_position', {})}")
            rospy.loginfo(f"  报告类型: {grid_data.get('report_type', 'N/A')}")
            
            # 检查动物类型是否完整
            expected_animals = ["Wolf", "Elephant", "Peacock", "Monkey", "Tiger"]
            animals = grid_data.get('animals', {})
            missing_animals = []
            for animal in expected_animals:
                if animal not in animals:
                    missing_animals.append(animal)
            
            if missing_animals:
                rospy.logwarn(f"❌ 缺少动物类型: {missing_animals}")
            else:
                rospy.loginfo("✅ 动物类型完整")
            
            rospy.loginfo("")
            
        except json.JSONDecodeError as e:
            rospy.logerr(f"❌ JSON解析错误: {e}")
            rospy.logerr(f"原始数据: {msg.data}")
    
    def run_test(self):
        """运行测试"""
        rospy.loginfo("开始测试新网格报告格式...")
        rospy.loginfo("等待60秒收集数据...")
        
        # 每20秒检查一次状态
        for i in range(3):
            rospy.sleep(20)
            rospy.loginfo(f"第{i+1}次状态检查: 收到 {self.report_count} 个报告")
        
        rospy.loginfo("=== 测试完成 ===")
        
        # 保持运行
        rate = rospy.Rate(1)  # 1Hz
        while not rospy.is_shutdown():
            rate.sleep()

def main():
    try:
        tester = NewGridFormatTester()
        tester.run_test()
    except rospy.ROSInterruptException:
        rospy.loginfo("测试被ROS中断")

if __name__ == "__main__":
    main()
