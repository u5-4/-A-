#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网格动物报告发布器
专门用于发布格式化的网格动物报告到 /grid_animal_report 话题
"""

import rospy
import json
import time
import math
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from yolov8_ros_msgs.msg import BoundingBoxes

class GridReportPublisher:
    def __init__(self):
        rospy.init_node('grid_report_publisher', anonymous=True)
        
        # 发布器
        self.grid_report_pub = rospy.Publisher('/grid_animal_report', String, queue_size=10)
        
        # 订阅器
        self.drone_pos_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.drone_position_callback)
        self.bboxes_sub = rospy.Subscriber('/bboxes_pub', BoundingBoxes, self.bboxes_callback)
        
        # 动物类别定义
        self.ANIMAL_CLASSES = {
            0: "Wolf",
            1: "Elephant", 
            2: "Peacock",
            3: "Monkey",
            4: "Tiger"
        }
        
        # 状态变量
        self.current_drone_x = 0.0
        self.current_drone_y = 0.0
        self.current_drone_z = 0.0
        self.animal_counters = {animal: 0 for animal in self.ANIMAL_CLASSES.values()}
        self.grid_animal_stats = {}
        self.counted_detections = set()
        self.last_report_time = 0
        self.report_interval = 5.0  # 每5秒发布一次报告
        
        rospy.loginfo("网格报告发布器已启动")
        rospy.loginfo("监控话题: /bboxes_pub, /mavros/local_position/pose")
        rospy.loginfo("发布话题: /grid_animal_report")
        rospy.loginfo(f"报告间隔: {self.report_interval}秒")
    
    def drone_position_callback(self, msg):
        """无人机位置回调"""
        self.current_drone_x = msg.pose.position.x
        self.current_drone_y = msg.pose.position.y
        self.current_drone_z = msg.pose.position.z
    
    def bboxes_callback(self, msg):
        """检测结果回调"""
        current_time = time.time()
        
        # 处理检测结果
        for bbox in msg.bounding_boxes:
            self.process_detection(bbox)
        
        # 定期发布报告
        if current_time - self.last_report_time >= self.report_interval:
            self.publish_grid_report()
            self.last_report_time = current_time
    
    def process_detection(self, bbox):
        """处理单个检测结果"""
        # 生成检测ID用于去重
        detection_id = f"{bbox.Class}_{int(bbox.xmin)}_{int(bbox.ymin)}_{int(bbox.xmax)}_{int(bbox.ymax)}"
        
        # 检查是否应该计数
        if self.should_count_detection(detection_id, bbox.probability):
            # 获取网格位置
            grid_pos = self.world_to_grid_position(self.current_drone_x, self.current_drone_y)
            
            # 更新计数
            animal_name = bbox.Class
            if animal_name in self.ANIMAL_CLASSES.values():
                self.animal_counters[animal_name] += 1
                
                # 更新网格统计
                if grid_pos not in self.grid_animal_stats:
                    self.grid_animal_stats[grid_pos] = {animal: 0 for animal in self.ANIMAL_CLASSES.values()}
                self.grid_animal_stats[grid_pos][animal_name] += 1
                
                rospy.loginfo(f"计数: {animal_name} (置信度: {bbox.probability:.2f}, 网格: {grid_pos})")
    
    def should_count_detection(self, detection_id, confidence):
        """判断是否应该计数"""
        # 检查置信度阈值
        if confidence < 0.6:
            return False
        
        # 检查是否已经计数过
        if detection_id in self.counted_detections:
            return False
        
        # 添加到已计数集合
        self.counted_detections.add(detection_id)
        return True
    
    def world_to_grid_position(self, world_x, world_y):
        """
        世界坐标转换为网格坐标（AXBX格式）
        
        坐标系定义：
        - 网格大小：0.5米一格
        - 网格范围：A1-A9 (列), B1-B7 (行)
        - 起飞点：A9B1 (0.0, 0.0)
        - 坐标系：X轴向前（机头方向），Y轴向左（垂直机头）
        """
        grid_size = 0.5
        grid_width = 9  # A1-A9
        grid_height = 7  # B1-B7
        
        # 正确的网格坐标计算
        # A列：从右到左（Y轴负方向），A9在原点(0,0)，A1在(4.0,0)
        # B行：从前到后（X轴正方向），B1在原点(0,0)，B7在(3.0,0)
        
        # Y坐标转换为A列（从右到左）
        col = int((4.0 - world_y) / grid_size)  # A9在(0,0)，A1在(4.0,0)
        
        # X坐标转换为B行（从前到后）
        row = int(world_x / grid_size)  # B1在(0,0)，B7在(3.0,0)
        
        # 边界检查
        if col < 0: col = 0
        if col >= grid_width: col = grid_width - 1
        if row < 0: row = 0
        if row >= grid_height: row = grid_height - 1
        
        # 转换为AXBX格式（确保格式正确）
        grid_pos = f"A{col+1}B{row+1}"
        
        # 调试信息
        rospy.logdebug(f"坐标转换: ({world_x:.2f}, {world_y:.2f}) -> 网格({col+1}, {row+1}) -> {grid_pos}")
        
        return grid_pos
    
    def publish_grid_report(self):
        """发布网格报告"""
        try:
            # 获取当前网格位置
            current_grid = self.world_to_grid_position(self.current_drone_x, self.current_drone_y)
            
            # 构建报告数据
            grid_report = {
                "grid_position": current_grid,
                "animals": {},
                "total_animals": self.animal_counters.copy(),
                "grid_total": 0,
                "mission_total": sum(self.animal_counters.values()),
                "drone_position": {
                    "x": round(self.current_drone_x, 2),
                    "y": round(self.current_drone_y, 2),
                    "z": round(self.current_drone_z, 2)
                },
                "report_type": "animal_change"
            }
            
            # 填充当前网格的动物统计
            if current_grid in self.grid_animal_stats:
                grid_report["animals"] = self.grid_animal_stats[current_grid]
                grid_report["grid_total"] = sum(self.grid_animal_stats[current_grid].values())
            else:
                # 如果没有当前网格的数据，填充所有动物为0
                grid_report["animals"] = {animal: 0 for animal in self.ANIMAL_CLASSES.values()}
            
            # 确保所有动物类型都在animals字段中
            for animal_name in self.ANIMAL_CLASSES.values():
                if animal_name not in grid_report["animals"]:
                    grid_report["animals"][animal_name] = 0
            
            # 发布报告
            grid_msg = String()
            grid_msg.data = json.dumps(grid_report, ensure_ascii=False, indent=2)
            self.grid_report_pub.publish(grid_msg)
            
            rospy.loginfo(f"发布网格报告: {current_grid}, 网格总数: {grid_report['grid_total']}, 任务总数: {grid_report['mission_total']}")
            
        except Exception as e:
            rospy.logerr(f"发布网格报告失败: {e}")
    
    def cleanup_old_detections(self):
        """清理旧的检测记录"""
        current_time = time.time()
        # 每60秒清理一次，或者如果检测记录过多时清理
        if len(self.counted_detections) > 1000:
            self.counted_detections.clear()
            rospy.loginfo("已清理旧的检测记录")
    
    def run(self):
        """运行发布器"""
        rate = rospy.Rate(1)  # 1Hz
        
        while not rospy.is_shutdown():
            # 定期清理
            self.cleanup_old_detections()
            
            # 即使没有检测结果，也定期发布报告
            current_time = time.time()
            if current_time - self.last_report_time >= self.report_interval:
                self.publish_grid_report()
                self.last_report_time = current_time
            
            rate.sleep()

def main():
    try:
        publisher = GridReportPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("网格报告发布器被中断")

if __name__ == "__main__":
    main() 
