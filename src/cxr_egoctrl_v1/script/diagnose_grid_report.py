#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断 /grid_animal_report 话题没有数据的原因
"""
import rospy
import time
import json
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
from yolov8_ros_msgs.msg import BoundingBoxes

class GridReportDiagnostic:
    def __init__(self):
        rospy.init_node('grid_report_diagnostic', anonymous=True)
        
        # 订阅相关话题
        self.bboxes_sub = rospy.Subscriber('/bboxes_pub', BoundingBoxes, self.bboxes_callback)
        self.animal_stats_sub = rospy.Subscriber('/animal_statistics', String, self.animal_stats_callback)
        self.grid_report_sub = rospy.Subscriber('/grid_animal_report', String, self.grid_report_callback)
        self.count_allow_sub = rospy.Subscriber('/count_allow', Bool, self.count_allow_callback)
        self.drone_pos_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.drone_pos_callback)
        
        # 状态变量
        self.bboxes_received = False
        self.animal_stats_received = False
        self.grid_report_received = False
        self.count_allowed = True  # 默认允许
        self.drone_position_received = False
        self.last_bboxes_time = None
        self.last_animal_stats_time = None
        self.last_grid_report_time = None
        self.bboxes_count = 0
        self.animal_stats_count = 0
        self.grid_report_count = 0
        
        # 检测统计
        self.detection_confidence_list = []
        self.detection_count = 0
        
        rospy.loginfo("网格报告诊断器已启动")
        rospy.loginfo("监控话题: /bboxes_pub, /animal_statistics, /grid_animal_report, /count_allow")
        
    def bboxes_callback(self, msg):
        """检测结果回调"""
        self.bboxes_received = True
        self.last_bboxes_time = time.time()
        self.bboxes_count += 1
        
        # 分析检测结果
        if msg.total_detections > 0:
            self.detection_count += msg.total_detections
            for bbox in msg.bounding_boxes:
                self.detection_confidence_list.append(bbox.probability)
                rospy.loginfo(f"检测到: {bbox.Class}, 置信度: {bbox.probability:.3f}")
        
        rospy.loginfo(f"收到检测结果: {msg.total_detections} 个目标, 平均置信度: {msg.average_confidence:.3f}")
    
    def animal_stats_callback(self, msg):
        """动物统计回调"""
        self.animal_stats_received = True
        self.last_animal_stats_time = time.time()
        self.animal_stats_count += 1
        
        try:
            stats = json.loads(msg.data)
            rospy.loginfo(f"收到动物统计: 总数={stats.get('total_count', 0)}, 动物={stats.get('animal_counts', {})}")
        except:
            rospy.logwarn("解析动物统计JSON失败")
    
    def grid_report_callback(self, msg):
        """网格报告回调"""
        self.grid_report_received = True
        self.last_grid_report_time = time.time()
        self.grid_report_count += 1
        
        try:
            grid_stats = json.loads(msg.data)
            rospy.loginfo(f"✅ 收到网格报告: {len(grid_stats)} 个网格有动物")
            for grid, animals in grid_stats.items():
                rospy.loginfo(f"  网格 {grid}: {animals}")
        except:
            rospy.logwarn("解析网格报告JSON失败")
    
    def count_allow_callback(self, msg):
        """计数允许信号回调"""
        self.count_allowed = msg.data
        rospy.loginfo(f"计数允许状态: {'允许' if msg.data else '禁止'}")
    
    def drone_pos_callback(self, msg):
        """无人机位置回调"""
        self.drone_position_received = True
        rospy.loginfo(f"无人机位置: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}, {msg.pose.position.z:.2f})")
    
    def check_topic_status(self):
        """检查话题状态"""
        rospy.loginfo("=== 话题状态检查 ===")
        
        # 检查检测结果话题
        if self.bboxes_received:
            time_since_bboxes = time.time() - self.last_bboxes_time if self.last_bboxes_time else 0
            rospy.loginfo(f"✅ /bboxes_pub: 正常, 接收次数: {self.bboxes_count}, 最后接收: {time_since_bboxes:.1f}秒前")
        else:
            rospy.logwarn("❌ /bboxes_pub: 未收到数据")
        
        # 检查动物统计话题
        if self.animal_stats_received:
            time_since_stats = time.time() - self.last_animal_stats_time if self.last_animal_stats_time else 0
            rospy.loginfo(f"✅ /animal_statistics: 正常, 接收次数: {self.animal_stats_count}, 最后接收: {time_since_stats:.1f}秒前")
        else:
            rospy.logwarn("❌ /animal_statistics: 未收到数据")
        
        # 检查网格报告话题
        if self.grid_report_received:
            time_since_grid = time.time() - self.last_grid_report_time if self.last_grid_report_time else 0
            rospy.loginfo(f"✅ /grid_animal_report: 正常, 接收次数: {self.grid_report_count}, 最后接收: {time_since_grid:.1f}秒前")
        else:
            rospy.logwarn("❌ /grid_animal_report: 未收到数据")
        
        # 检查计数允许状态
        rospy.loginfo(f"计数允许状态: {'✅ 允许' if self.count_allowed else '❌ 禁止'}")
        
        # 检查无人机位置
        if self.drone_position_received:
            rospy.loginfo("✅ 无人机位置: 正常接收")
        else:
            rospy.logwarn("❌ 无人机位置: 未收到数据")
    
    def analyze_detection_conditions(self):
        """分析检测条件"""
        rospy.loginfo("=== 检测条件分析 ===")
        
        if not self.detection_confidence_list:
            rospy.logwarn("❌ 没有检测到任何动物")
            return
        
        # 分析置信度分布
        confidences = sorted(self.detection_confidence_list)
        min_conf = min(confidences)
        max_conf = max(confidences)
        avg_conf = sum(confidences) / len(confidences)
        
        rospy.loginfo(f"检测统计: 总数={len(confidences)}, 置信度范围=[{min_conf:.3f}, {max_conf:.3f}], 平均={avg_conf:.3f}")
        
        # 检查置信度阈值问题
        count_threshold = 0.9
        above_threshold = [c for c in confidences if c >= count_threshold]
        rospy.loginfo(f"置信度≥{count_threshold}: {len(above_threshold)}/{len(confidences)} ({len(above_threshold)/len(confidences)*100:.1f}%)")
        
        if len(above_threshold) == 0:
            rospy.logwarn(f"❌ 所有检测的置信度都低于计数阈值 {count_threshold}")
            rospy.logwarn("建议: 降低置信度阈值或改善检测模型")
        
        # 检查计数允许问题
        if not self.count_allowed:
            rospy.logwarn("❌ 主控制器禁止计数，这是网格报告为空的主要原因")
            rospy.logwarn("建议: 检查主控制器是否发送了允许计数的信号")
    
    def check_potential_issues(self):
        """检查潜在问题"""
        rospy.loginfo("=== 潜在问题分析 ===")
        
        issues = []
        
        # 问题1: 没有检测到动物
        if self.detection_count == 0:
            issues.append("没有检测到任何动物")
        
        # 问题2: 置信度太低
        if self.detection_confidence_list and max(self.detection_confidence_list) < 0.9:
            issues.append("所有检测的置信度都低于计数阈值0.9")
        
        # 问题3: 计数被禁止
        if not self.count_allowed:
            issues.append("主控制器禁止计数")
        
        # 问题4: 没有无人机位置
        if not self.drone_position_received:
            issues.append("没有收到无人机位置信息，无法计算网格坐标")
        
        # 问题5: 检测结果正常但没有网格报告
        if self.bboxes_received and self.animal_stats_received and not self.grid_report_received:
            issues.append("有检测结果和统计信息，但没有网格报告（可能是网格坐标计算问题）")
        
        if issues:
            rospy.logwarn("发现以下问题:")
            for i, issue in enumerate(issues, 1):
                rospy.logwarn(f"  {i}. {issue}")
        else:
            rospy.loginfo("✅ 未发现明显问题")
    
    def run_diagnostic(self):
        """运行诊断"""
        rospy.loginfo("开始诊断 /grid_animal_report 话题问题...")
        rospy.loginfo("请确保YOLO检测系统正在运行")
        rospy.loginfo("按Ctrl+C停止诊断")
        
        try:
            rate = rospy.Rate(1)  # 1Hz
            start_time = time.time()
            
            while not rospy.is_shutdown():
                elapsed = time.time() - start_time
                
                if elapsed > 30:  # 30秒后开始分析
                    rospy.loginfo(f"\n=== 诊断报告 (运行{elapsed:.0f}秒) ===")
                    self.check_topic_status()
                    self.analyze_detection_conditions()
                    self.check_potential_issues()
                    
                    # 每60秒重复一次分析
                    if int(elapsed) % 60 == 0:
                        rospy.loginfo("\n" + "="*50)
                
                rate.sleep()
                
        except KeyboardInterrupt:
            rospy.loginfo("诊断被用户中断")
        except Exception as e:
            rospy.logerr(f"诊断过程中出现错误: {e}")

def main():
    try:
        diagnostic = GridReportDiagnostic()
        diagnostic.run_diagnostic()
    except rospy.ROSInterruptException:
        rospy.loginfo("诊断被ROS中断")

if __name__ == "__main__":
    main() 
