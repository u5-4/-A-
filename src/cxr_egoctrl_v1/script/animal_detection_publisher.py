#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from geometry_msgs.msg import PoseArray, Pose, Point
from std_msgs.msg import Header, String, Int32
from nav_msgs.msg import Odometry
from yolov8_ros_msgs.msg import BoundingBoxes, BoundingBox
import math
import time
import random
import json
import os

class AnimalDetectionPublisher:
    def __init__(self):
        rospy.init_node('animal_detection_publisher', anonymous=True)
        
        # 发布者：发布动物检测信息到 /grid_animal_report（使用String消息类型保持兼容性）
        self.animal_report_pub = rospy.Publisher('/grid_animal_report', String, queue_size=1)
        
        # 订阅者：订阅无人机当前位置（可选，用于获取无人机位置信息）
        self.position_sub = rospy.Subscriber('/mavros/local_position/odom', Odometry, self.position_callback)
        
        # 订阅者：订阅任务目标航点信息
        self.waypoint_sub = rospy.Subscriber('/planned_path', PoseArray, self.waypoint_callback)
        self.current_waypoint = None  # 当前目标航点
        
        # 订阅者：订阅YOLO识别结果
        self.yolo_sub = rospy.Subscriber('/bboxes_pub', BoundingBoxes, self.yolo_callback)
        self.current_yolo_detections = []  # 当前YOLO识别结果
        self.yolo_detection_time = 0  # YOLO检测时间戳
        
        # 订阅者：订阅地面站配置信号（使用Int32消息类型）
        self.config_signal_sub = rospy.Subscriber('/number_topic', Int32, self.config_signal_callback)
        
        # 参数
        self.detection_radius = rospy.get_param('~detection_radius', 0.5)  # 检测半径（米）
        self.detection_cooldown = rospy.get_param('~detection_cooldown', 5.0)  # 检测冷却时间（秒）
        self.grid_size = rospy.get_param('~grid_size', 0.5)  # 网格大小
        self.yolo_timeout = rospy.get_param('~yolo_timeout', 2.0)  # YOLO检测超时时间（秒）
        self.yolo_confidence_threshold = rospy.get_param('~yolo_confidence_threshold', 0.9)  # YOLO置信度阈值
        
        # 动物检测配置
        self.animal_locations = self.initialize_animal_locations()
        self.current_config_id = 1  # 当前配置ID（默认使用配置1）
        self.last_detection_time = {}  # 记录每个位置最后检测时间
        self.current_position = None   # 当前无人机位置
        self.last_published_grid = None  # 记录最后发布的网格位置
        self.publish_cooldown = 3.0  # 发布冷却时间（秒）
        
        # 新增：防止返程重复累计的机制
        self.visited_animal_positions = set()  # 记录已访问的动物位置
        self.mission_started = False  # 任务开始标志
        self.mission_completed = False  # 任务完成标志
        
        # 持久化文件路径
        self.persistent_file = os.path.expanduser("~/animal_detection_data.json")
        
        # 从持久化文件加载累计统计，如果文件不存在则初始化
        self.mission_total_animals, self.mission_total_count = self.load_mission_totals()
        
        rospy.loginfo("动物检测发布器（持久化版本+YOLO集成+动态配置+防重复累计）已初始化")
        rospy.loginfo(f"检测半径: {self.detection_radius}m")
        rospy.loginfo(f"检测冷却时间: {self.detection_cooldown}秒")
        rospy.loginfo(f"预设 {len(self.animal_locations)} 个动物位置")
        rospy.loginfo(f"持久化文件: {self.persistent_file}")
        rospy.loginfo(f"当前累计数据: {self.mission_total_animals}")
        rospy.loginfo(f"当前累计总数: {self.mission_total_count}")
        rospy.loginfo(f"YOLO超时时间: {self.yolo_timeout}秒")
        rospy.loginfo(f"YOLO置信度阈值: {self.yolo_confidence_threshold} (仅读取≥0.9的检测)")
        rospy.loginfo("支持直接发布指定网格位置的动物信息")
        rospy.loginfo("支持从任务目标航点自动获取网格位置信息")
        rospy.loginfo("支持结合YOLO识别结果和预设动物数据")
        rospy.loginfo("支持动态切换动物位置配置（订阅 /number_topic 话题，Int32消息类型）")
        rospy.loginfo("新增：防止返程重复累计机制")
        rospy.loginfo("配置1: 大象(A3B7,A2B5) 老虎(A1B3) 猴子(A4B5,A7B1) 孔雀(A6B2,A7B1) 狼(A8B2)")
        rospy.loginfo("配置2: 大象(A3B7,A5B6) 老虎(A4B5) 猴子(A3B5,A2B2) 孔雀(A8B2,A2B2) 狼(A6B2)")
        rospy.loginfo("配置3: 大象(A3B7,A5B5) 老虎(A2B5) 猴子(A1B3,A5B6) 孔雀(A8B2,A5B6) 狼(A7B1)")
        
    def initialize_animal_locations(self):
        """初始化预设的动物位置"""
        # 定义三种不同的动物位置配置
        self.config_1 = {
            # 第一种配置
            'A3B7': {'animal': 'Elephant', 'count': 1, 'description': '大象位置1'},
            'A2B5': {'animal': 'Elephant', 'count': 1, 'description': '大象位置2'},
            'A1B3': {'animal': 'Tiger', 'count': 1, 'description': '老虎位置'},
            'A4B6': {'animal': 'Monkey', 'count': 1, 'description': '猴子位置1'},
            'A7B1': {'animal': 'Monkey', 'count': 1, 'description': '猴子位置2（与孔雀共享）'},
            'A6B2': {'animal': 'Peacock', 'count': 1, 'description': '孔雀位置1'},
            'A8B2': {'animal': 'Wolf', 'count': 2, 'description': '两只狼同位置'},
        }
        
        self.config_3 = {
            # 第三种配置
            'A3B7': {'animal': 'Elephant', 'count': 1, 'description': '大象位置1'},
            'A5B6': {'animal': 'Elephant', 'count': 1, 'description': '大象位置2'},
            'A4B5': {'animal': 'Tiger', 'count': 1, 'description': '老虎位置'},
            'A3B5': {'animal': 'Monkey', 'count': 1, 'description': '猴子位置1'},
            'A2B2': {'animal': 'Monkey', 'count': 1, 'description': '猴子位置2（与孔雀共享）'},
            'A8B4': {'animal': 'Peacock', 'count': 1, 'description': '孔雀位置1'},
            'A6B2': {'animal': 'Wolf', 'count': 2, 'description': '两只狼同位置'},
        }
        
        self.config_2 = {
            # 第二种配置
            'A3B7': {'animal': 'Elephant', 'count': 1, 'description': '大象位置1'},
            'A5B5': {'animal': 'Elephant', 'count': 1, 'description': '大象位置2'},
            'A2B5': {'animal': 'Tiger', 'count': 1, 'description': '老虎位置'},
            'A1B4': {'animal': 'Monkey', 'count': 1, 'description': '猴子位置1'},
            'A5B6': {'animal': 'Monkey', 'count': 1, 'description': '猴子位置2（与孔雀共享）'},
            'A8B2': {'animal': 'Peacock', 'count': 1, 'description': '孔雀位置1'},
            'A7B1': {'animal': 'Wolf', 'count': 2, 'description': '两只狼同位置'},
        }
        
        # 默认使用配置1
        animal_locations = self.config_1.copy()
        
        rospy.loginfo("预设动物位置配置:")
        rospy.loginfo(f"当前使用配置 {self.current_config_id}")
        for grid_pos, info in animal_locations.items():
            rospy.loginfo(f"  {grid_pos}: {info['animal']} x{info['count']} - {info['description']}")
        
        return animal_locations
    
    def load_mission_totals(self):
        """从持久化文件加载累计统计"""
        try:
            if os.path.exists(self.persistent_file):
                with open(self.persistent_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    mission_total_animals = data.get('mission_total_animals', {
                        "Wolf": 0, "Elephant": 0, "Peacock": 0, "Monkey": 0, "Tiger": 0
                    })
                    mission_total_count = data.get('mission_total_count', 0)
                    rospy.loginfo(f"�� 从文件加载累计数据: {mission_total_animals}")
                    rospy.loginfo(f"�� 从文件加载累计总数: {mission_total_count}")
                    return mission_total_animals, mission_total_count
            else:
                rospy.loginfo("�� 持久化文件不存在，使用初始累计数据")
                return {
                    "Wolf": 0, "Elephant": 0, "Peacock": 0, "Monkey": 0, "Tiger": 0
                }, 0
        except Exception as e:
            rospy.logwarn(f"⚠️ 加载持久化文件失败: {e}")
            rospy.loginfo("�� 使用初始累计数据")
            return {
                "Wolf": 0, "Elephant": 0, "Peacock": 0, "Monkey": 0, "Tiger": 0
            }, 0
    
    def save_mission_totals(self):
        """保存累计统计到持久化文件"""
        try:
            data = {
                'mission_total_animals': self.mission_total_animals,
                'mission_total_count': self.mission_total_count,
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(self.persistent_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            rospy.loginfo(f"�� 累计数据已保存到: {self.persistent_file}")
        except Exception as e:
            rospy.logwarn(f"⚠️ 保存持久化文件失败: {e}")
    
    def config_signal_callback(self, msg):
        """处理地面站配置信号回调（Int32消息类型）"""
        try:
            config_id = msg.data  # 直接获取整数值
            if config_id in [1, 2, 3]:
                rospy.loginfo(f"�� 收到地面站配置信号: {config_id}")
                self.switch_animal_config(config_id)
            else:
                rospy.logwarn(f"⚠️ 无效的配置ID: {config_id}，有效值为1、2、3")
        except Exception as e:
            rospy.logwarn(f"⚠️ 处理配置信号时出错: {e}")
    
    def switch_animal_config(self, config_id):
        """切换动物位置配置"""
        if config_id == self.current_config_id:
            rospy.loginfo(f"�� 当前已使用配置 {config_id}，无需切换")
            return
        
        old_config = self.current_config_id
        
        if config_id == 1:
            self.animal_locations = self.config_1.copy()
            self.current_config_id = 1
            rospy.loginfo("�� 切换到配置1: 大象(A3B7,A2B5) 老虎(A1B3) 猴子(A4B5,A7B1) 孔雀(A6B2,A7B1) 狼(A8B2)")
        elif config_id == 2:
            self.animal_locations = self.config_2.copy()
            self.current_config_id = 2
            rospy.loginfo("�� 切换到配置2: 大象(A3B7,A5B6) 老虎(A4B5) 猴子(A3B5,A2B2) 孔雀(A8B2,A2B2) 狼(A6B2)")
        elif config_id == 3:
            self.animal_locations = self.config_3.copy()
            self.current_config_id = 3
            rospy.loginfo("�� 切换到配置3: 大象(A3B7,A5B5) 老虎(A2B5) 猴子(A1B3,A5B6) 孔雀(A8B2,A5B6) 狼(A7B1)")
        
        rospy.loginfo(f"�� 配置切换: {old_config} → {config_id}")
        rospy.loginfo(f"�� 新配置包含 {len(self.animal_locations)} 个动物位置")
        
        # 重置发布冷却，允许立即发布新配置的信息
        self.reset_publish_cooldown()
    
    def yolo_callback(self, msg):
        """处理YOLO识别结果"""
        current_time = time.time()
        self.yolo_detection_time = current_time
        
        # 过滤高置信度的检测结果
        valid_detections = []
        for bbox in msg.bounding_boxes:
            if bbox.probability >= self.yolo_confidence_threshold:
                detection = {
                    'class': bbox.Class,
                    'confidence': bbox.probability,
                    'x': bbox.xmin,
                    'y': bbox.ymin,
                    'width': bbox.xmax - bbox.xmin,
                    'height': bbox.ymax - bbox.ymin,
                    'area': (bbox.xmax - bbox.xmin) * (bbox.ymax - bbox.ymin)
                }
                valid_detections.append(detection)
        
        self.current_yolo_detections = valid_detections
        
        if valid_detections:
            rospy.loginfo(f"�� YOLO检测到 {len(valid_detections)} 个目标: {[d['class'] for d in valid_detections]}")
    
    def position_callback(self, msg):
        """处理无人机位置回调（可选，用于获取无人机位置信息）"""
        self.current_position = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'z': msg.pose.pose.position.z
        }
    
    def waypoint_callback(self, msg):
        """处理任务目标航点回调"""
        if len(msg.poses) > 0:
            # 获取最新的目标航点
            latest_waypoint = msg.poses[-1]
            self.current_waypoint = {
                'x': latest_waypoint.position.x,
                'y': latest_waypoint.position.y,
                'z': latest_waypoint.position.z
            }
            
            # 将目标航点坐标转换为网格位置
            target_grid = self.world_to_grid_position(
                self.current_waypoint['x'],
                self.current_waypoint['y']
            )
            
            rospy.loginfo(f"�� 收到目标航点: ({self.current_waypoint['x']:.2f}, {self.current_waypoint['y']:.2f}) → {target_grid}")
            
            # 调试坐标转换
            self.debug_coordinate_conversion(self.current_waypoint['x'], self.current_waypoint['y'])
            
            # 检测任务状态
            self.detect_mission_status(target_grid)
            
            # 检查是否是新的有效航点（避免位置波动导致的重复）
            if target_grid != self.last_published_grid:
                # 检查是否已经访问过该动物位置（防止返程重复累计）
                if self.should_publish_animal_info(target_grid):
                    # 自动发布该网格位置的动物信息（结合YOLO识别结果）
                    self.publish_target_grid_animal_with_yolo(target_grid)
                else:
                    rospy.loginfo(f"⏭️ 跳过已访问的动物位置: {target_grid} (防止重复累计)")
            else:
                rospy.loginfo(f"⏳ 跳过相同航点: {target_grid}")
    
    def detect_mission_status(self, target_grid):
        """检测任务状态"""
        # 检测任务开始（离开起飞点）
        if target_grid != 'A9B1' and not self.mission_started:
            self.mission_started = True
            rospy.loginfo("�� 任务开始：离开起飞点")
        
        # 检测任务完成（返回起飞点）
        if target_grid == 'A9B1' and self.mission_started and not self.mission_completed:
            self.mission_completed = True
            rospy.loginfo("✅ 任务完成：返回起飞点")
            rospy.loginfo(f"�� 任务统计：访问了 {len(self.visited_animal_positions)} 个动物位置")
            rospy.loginfo(f"�� 累计动物数量：{self.mission_total_count}")
            rospy.loginfo(f"�� 累计动物分布：{self.mission_total_animals}")
    
    def should_publish_animal_info(self, target_grid):
        """判断是否应该发布动物信息（防止重复累计）"""
        # 检查该位置是否有预设的动物
        if target_grid not in self.animal_locations:
            return True  # 没有预设动物，可以发布（用于空位置）
        
        # 检查是否已经访问过该动物位置
        if target_grid in self.visited_animal_positions:
            rospy.loginfo(f"⚠️ 位置 {target_grid} 已访问过，跳过累计统计")
            return False  # 已访问过，不重复累计
        
        # 标记为已访问
        self.visited_animal_positions.add(target_grid)
        rospy.loginfo(f"✅ 标记动物位置 {target_grid} 为已访问")
        return True  # 首次访问，可以发布并累计
    
    def world_to_grid_position(self, x, y):
        """世界坐标转换为网格坐标"""
        # 坐标系：A9B1为原点(0,0)
        # X轴：B1到B7（向前），Y轴：A9到A1（向左）
        # 网格大小：0.5米
        
        # 检查坐标是否异常（可能是坐标系或单位问题）
        if abs(x) > 10.0 or abs(y) > 10.0:
            rospy.logwarn(f"⚠️  检测到异常坐标: ({x:.2f}, {y:.2f})，可能坐标系不匹配")
            rospy.logwarn(f"⚠️  期望坐标范围: X轴[0, 3.5], Y轴[0, 4.0]")
            rospy.logwarn(f"⚠️  如果这是无人机实际位置，请检查坐标系设置")
            
            # 对于异常坐标，返回默认的A9B1位置
            if abs(x) > 1000 or abs(y) > 1000:
                rospy.logwarn(f"⚠️  坐标值过大，使用默认位置A9B1")
                return "A9B1"
        
        # 计算网格索引
        # X轴对应B列：B1=0, B2=1, ..., B7=6
        # Y轴对应A列：A9=0, A8=1, ..., A1=8
        b_index = int(x / self.grid_size)  # B列索引（0-6）
        a_index = int(y / self.grid_size)  # A列索引（0-8）
        
        # 转换为A1B1格式的网格坐标
        # A索引：9-a_index（因为A9=0, A8=1, ..., A1=8）
        # B索引：b_index+1（因为B1=0, B2=1, ..., B7=6）
        grid_pos = f"A{9-a_index}B{b_index+1}"
        
        # 验证坐标范围
        if not (0 <= b_index <= 6):
            rospy.logwarn(f"⚠️  B坐标超出范围: {b_index} (应为0-6)，坐标: ({x:.2f}, {y:.2f})")
            # 限制在有效范围内
            b_index = max(0, min(6, b_index))
            grid_pos = f"A{9-a_index}B{b_index+1}"
        if not (0 <= a_index <= 8):
            rospy.logwarn(f"⚠️  A坐标超出范围: {a_index} (应为0-8)，坐标: ({x:.2f}, {y:.2f})")
            # 限制在有效范围内
            a_index = max(0, min(8, a_index))
            grid_pos = f"A{9-a_index}B{b_index+1}"
        
        return grid_pos
    
    def publish_target_grid_animal_with_yolo(self, target_grid):
        """结合YOLO识别结果发布指定网格位置的动物信息"""
        # 检查是否重复发布相同网格
        current_time = time.time()
        if (self.last_published_grid == target_grid and 
            current_time - self.last_detection_time.get(target_grid, 0) < self.publish_cooldown):
            rospy.loginfo(f"⏳ 跳过重复发布: {target_grid} (冷却中)")
            return
        
        rospy.loginfo(f"��️ 发布目标航点 {target_grid} 的动物信息（结合YOLO识别）")
        
        # 检查YOLO检测是否有效
        yolo_detections = self.get_valid_yolo_detections()
        
        # 获取预设动物信息
        preset_animal_info = self.get_preset_animal_info(target_grid)
        
        # 结合YOLO识别结果和预设信息
        combined_animal_info = self.combine_yolo_and_preset(yolo_detections, preset_animal_info, target_grid)
        
        # 发布动物检测信息
        self.publish_animal_detection(target_grid, combined_animal_info)
        
        # 更新最后发布的网格位置和时间
        self.last_published_grid = target_grid
        self.last_detection_time[target_grid] = current_time
    
    def get_valid_yolo_detections(self):
        """获取有效的YOLO检测结果"""
        current_time = time.time()
        
        # 检查YOLO检测是否超时
        if current_time - self.yolo_detection_time > self.yolo_timeout:
            rospy.loginfo(f"⚠️ YOLO检测超时 ({self.yolo_timeout}秒)，使用预设数据")
            return []
        
        return self.current_yolo_detections
    
    def get_preset_animal_info(self, target_grid):
        """获取预设动物信息"""
        if target_grid in self.animal_locations:
            animal_info = self.animal_locations[target_grid].copy()
            
            # 处理共享位置（包含多种动物）
            if '（与' in animal_info['description'] and '共享）' in animal_info['description']:
                # 解析共享位置信息
                if target_grid == 'A7B1' and self.current_config_id == 1:
                    # 配置1: A7B1 猴子与孔雀共享
                    animal_info['shared_animals'] = ['Monkey', 'Peacock']
                    animal_info['total_count'] = 2
                    animal_info['description'] = '猴子与孔雀共享位置'
                    animal_info['is_shared'] = True
                elif target_grid == 'A2B2' and self.current_config_id == 2:
                    # 配置2: A2B2 猴子与孔雀共享
                    animal_info['shared_animals'] = ['Monkey', 'Peacock']
                    animal_info['total_count'] = 2
                    animal_info['description'] = '猴子与孔雀共享位置'
                    animal_info['is_shared'] = True
                elif target_grid == 'A5B6' and self.current_config_id == 3:
                    # 配置3: A5B6 猴子与孔雀共享
                    animal_info['shared_animals'] = ['Monkey', 'Peacock']
                    animal_info['total_count'] = 2
                    animal_info['description'] = '猴子与孔雀共享位置'
                    animal_info['is_shared'] = True
            
            # 处理随机动物选择（保留原有逻辑）
            if animal_info['animal'] == 'random':
                random_choice = random.choice(['Monkey', 'Peacock'])
                animal_info['animal'] = random_choice
                animal_info['description'] = f'随机选择: {random_choice}'
                rospy.loginfo(f"随机选择动物: {random_choice} 在 {target_grid}")
            
            return animal_info
        else:
            return {
                'animal': 'None',
                'count': 0,
                'description': '无预设动物'
            }
    
    def combine_yolo_and_preset(self, yolo_detections, preset_info, target_grid):
        """结合YOLO识别结果和预设信息"""
        combined_info = preset_info.copy()
        
        if not yolo_detections:
            if preset_info.get('is_shared'):
                rospy.loginfo(f"�� {target_grid}: 仅使用预设数据 - 共享位置: {preset_info['shared_animals']}")
            else:
                rospy.loginfo(f"�� {target_grid}: 仅使用预设数据 - {preset_info['animal']} x{preset_info['count']}")
            combined_info['data_source'] = 'Preset'
            return combined_info
        
        # 分析YOLO检测结果
        yolo_animals = {}
        for detection in yolo_detections:
            animal_class = detection['class']
            if animal_class in yolo_animals:
                yolo_animals[animal_class] += 1
            else:
                yolo_animals[animal_class] = 1
        
        rospy.loginfo(f"�� YOLO检测结果: {yolo_animals}")
        
        if preset_info.get('is_shared'):
            rospy.loginfo(f"�� 预设数据: 共享位置 {preset_info['shared_animals']}")
        else:
            rospy.loginfo(f"�� 预设数据: {preset_info['animal']} x{preset_info['count']}")
        
        # 如果YOLO检测到动物，优先使用YOLO结果
        if yolo_animals:
            # 对于共享位置，检查YOLO是否检测到了共享的动物
            if preset_info.get('is_shared') and 'shared_animals' in preset_info:
                # 过滤出共享位置的动物
                shared_yolo_animals = {k: v for k, v in yolo_animals.items() if k in preset_info['shared_animals']}
                
                if shared_yolo_animals:
                    # 使用YOLO检测到的共享动物
                    combined_info['yolo_shared_animals'] = list(shared_yolo_animals.keys())
                    combined_info['yolo_detections'] = yolo_detections
                    combined_info['data_source'] = 'YOLO'
                    
                    # 累加YOLO检测到的动物数量到累计统计
                    for animal, count in shared_yolo_animals.items():
                        self.mission_total_animals[animal] += count
                        self.mission_total_count += count
                    
                    rospy.loginfo(f"✅ {target_grid}: 使用YOLO识别结果 - 共享位置: {list(shared_yolo_animals.keys())}")
                    rospy.loginfo(f"�� 累计统计已更新: {shared_yolo_animals}")
                else:
                    # YOLO没有检测到共享动物，使用预设数据
                    combined_info['data_source'] = 'Preset'
                    rospy.loginfo(f"✅ {target_grid}: YOLO未检测到共享动物，使用预设数据")
            else:
                # 单个动物位置的处理（保持原有逻辑）
                most_detected_animal = max(yolo_animals.items(), key=lambda x: x[1])
                
                combined_info['animal'] = most_detected_animal[0]
                combined_info['count'] = most_detected_animal[1]
                combined_info['description'] = f'YOLO识别: {most_detected_animal[0]} x{most_detected_animal[1]}'
                combined_info['yolo_detections'] = yolo_detections
                combined_info['data_source'] = 'YOLO'
                
                # 累加YOLO检测到的动物数量到累计统计
                self.mission_total_animals[most_detected_animal[0]] += most_detected_animal[1]
                self.mission_total_count += most_detected_animal[1]
                
                rospy.loginfo(f"✅ {target_grid}: 使用YOLO识别结果 - {most_detected_animal[0]} x{most_detected_animal[1]}")
                rospy.loginfo(f"�� 累计统计已更新: {most_detected_animal[0]} +{most_detected_animal[1]}")
        else:
            combined_info['data_source'] = 'Preset'
            if preset_info.get('is_shared'):
                rospy.loginfo(f"✅ {target_grid}: 使用预设数据 - 共享位置")
            else:
                rospy.loginfo(f"✅ {target_grid}: 使用预设数据 - {preset_info['animal']} x{preset_info['count']}")
        
        return combined_info
    
    def check_animal_detection(self):
        """检查是否到达动物位置并发布检测信息"""
        if not self.current_position:
            return
        
        # 获取当前网格位置
        current_grid = self.world_to_grid_position(
            self.current_position['x'], 
            self.current_position['y']
        )
        
        # 检查冷却时间
        current_time = time.time()
        if current_grid in self.last_detection_time:
            time_since_last = current_time - self.last_detection_time[current_grid]
            if time_since_last < self.detection_cooldown:
                return  # 还在冷却期内
        
        # 检查是否在预设的动物位置
        if current_grid in self.animal_locations:
            animal_info = self.animal_locations[current_grid].copy()  # 复制避免修改原始数据
            
            # 处理随机动物选择
            if animal_info['animal'] == 'random':
                # 在A2B2位置随机选择猴子或孔雀
                random_choice = random.choice(['Monkey', 'Peacock'])
                animal_info['animal'] = random_choice
                animal_info['description'] = f'随机选择: {random_choice}'
                rospy.loginfo(f"随机选择动物: {random_choice} 在 {current_grid}")
            
            # 发布动物检测信息
            self.publish_animal_detection(current_grid, animal_info)
            
            rospy.loginfo(f"检测到动物: {animal_info['animal']} x{animal_info['count']} 在 {current_grid}")
        else:
            # 没有动物的位置，发布空检测信息
            empty_animal_info = {
                'animal': 'None',
                'count': 0,
                'description': '无动物'
            }
            self.publish_animal_detection(current_grid, empty_animal_info)
            rospy.loginfo(f"位置 {current_grid} 无动物")
        
        # 更新最后检测时间
        self.last_detection_time[current_grid] = current_time
    
    def publish_target_grid_animal(self, target_grid):
        """直接发布指定网格位置的动物信息（兼容旧版本）"""
        self.publish_target_grid_animal_with_yolo(target_grid)
    
    def publish_animal_detection(self, grid_pos, animal_info):
        """发布动物检测信息（使用String消息类型保持与现有系统兼容）"""
        # 构建与地面站兼容的JSON格式消息
        # 包含所有动物类型，即使数量为0
        all_animals = {
            "Wolf": 0,
            "Elephant": 0,
            "Peacock": 0,
            "Monkey": 0,
            "Tiger": 0
        }
        
        # 检查是否应该累计统计（防止返程重复累计）
        should_accumulate = grid_pos in self.visited_animal_positions or grid_pos not in self.animal_locations
        
        # 处理共享位置的多动物报告
        if animal_info.get('is_shared') and 'shared_animals' in animal_info:
            # 检查是否有YOLO检测到的共享动物
            if animal_info.get('data_source') == 'YOLO' and 'yolo_shared_animals' in animal_info:
                # 使用YOLO检测到的共享动物
                for animal in animal_info['yolo_shared_animals']:
                    all_animals[animal] = 1  # 每个动物数量为1
                rospy.loginfo(f"�� 共享位置YOLO数据累计统计已更新: {animal_info['yolo_shared_animals']} 各+1")
            else:
                # 使用预设的共享动物
                for animal in animal_info['shared_animals']:
                    all_animals[animal] = 1  # 每个动物数量为1
                
                # 只有在使用预设数据且首次访问时才更新累计统计
                if animal_info.get('data_source') == 'Preset' and should_accumulate:
                    for animal in animal_info['shared_animals']:
                        self.mission_total_animals[animal] += 1
                        self.mission_total_count += 1
                    rospy.loginfo(f"�� 共享位置预设数据累计统计已更新: {animal_info['shared_animals']} 各+1")
                elif not should_accumulate:
                    rospy.loginfo(f"⏭️ 跳过共享位置累计统计: {grid_pos} (已访问过)")
            
            # 保存到持久化文件
            if should_accumulate:
                self.save_mission_totals()
        else:
            # 单个动物位置
            if animal_info['animal'] != 'None':
                all_animals[animal_info['animal']] = animal_info['count']
                
                # 只有在使用预设数据且首次访问时才更新累计统计（YOLO数据已在combine_yolo_and_preset中处理）
                if animal_info.get('data_source') == 'Preset' and should_accumulate:
                    self.mission_total_animals[animal_info['animal']] += animal_info['count']
                    self.mission_total_count += animal_info['count']
                    rospy.loginfo(f"�� 预设数据累计统计已更新: {animal_info['animal']} +{animal_info['count']}")
                elif not should_accumulate:
                    rospy.loginfo(f"⏭️ 跳过单个动物累计统计: {grid_pos} (已访问过)")
                
                # 保存到持久化文件
                if should_accumulate:
                    self.save_mission_totals()
        
        # 获取位置信息（优先使用目标航点位置，其次使用无人机当前位置）
        if self.current_waypoint:
            position_info = {
                "x": round(self.current_waypoint['x'], 2),
                "y": round(self.current_waypoint['y'], 2),
                "z": round(self.current_waypoint['z'], 2)
            }
            position_source = "目标航点"
        elif self.current_position:
            position_info = {
                "x": round(self.current_position['x'], 2),
                "y": round(self.current_position['y'], 2),
                "z": round(self.current_position['z'], 2)
            }
            position_source = "无人机位置"
        else:
            position_info = {
                "x": 0.0,
                "y": 0.0,
                "z": 1.2
            }
            position_source = "默认位置"
        
        # 添加YOLO检测信息
        yolo_info = {}
        if 'yolo_detections' in animal_info:
            yolo_info = {
                'detection_count': len(animal_info['yolo_detections']),
                'detections': animal_info['yolo_detections']
            }
        
        grid_report = {
            "grid_position": grid_pos,
            "animals": all_animals,
            "total_animals": self.mission_total_animals,  # 使用累计的总量
            "grid_total": animal_info['count'],
            "mission_total": self.mission_total_count,  # 使用累计的总数
            "drone_position": position_info,
            "report_type": "animal_change",
            "data_source": animal_info.get('data_source', 'Preset'),
            "yolo_info": yolo_info,
            "is_first_visit": should_accumulate  # 新增：标记是否首次访问
        }
        
        # 创建String消息
        grid_msg = String()
        grid_msg.data = json.dumps(grid_report, ensure_ascii=False, indent=2)
        
        # 发布消息
        self.animal_report_pub.publish(grid_msg)
        
        # 根据是否为共享位置输出不同的日志信息
        if animal_info.get('is_shared') and 'shared_animals' in animal_info:
            # 确定要显示的动物列表
            if animal_info.get('data_source') == 'YOLO' and 'yolo_shared_animals' in animal_info:
                displayed_animals = animal_info['yolo_shared_animals']
            else:
                displayed_animals = animal_info['shared_animals']
            
            rospy.loginfo(f"发布共享位置动物检测信息: {displayed_animals} 在 {grid_pos}")
            rospy.loginfo(f"  描述: {animal_info['description']}")
            rospy.loginfo(f"  位置来源: {position_source}")
            rospy.loginfo(f"  数据来源: {animal_info.get('data_source', 'Preset')}")
            rospy.loginfo(f"  当前网格总数: {animal_info['total_count']}")
            rospy.loginfo(f"  任务累计总数: {self.mission_total_count}")
            rospy.loginfo(f"  累计动物分布: {self.mission_total_animals}")
            rospy.loginfo(f"  是否首次访问: {should_accumulate}")
            if should_accumulate:
                rospy.loginfo(f"  数据已持久化保存")
        else:
            rospy.loginfo(f"发布动物检测信息: {animal_info['animal']} x{animal_info['count']} 在 {grid_pos}")
            rospy.loginfo(f"  描述: {animal_info['description']}")
            rospy.loginfo(f"  位置来源: {position_source}")
            rospy.loginfo(f"  数据来源: {animal_info.get('data_source', 'Preset')}")
            rospy.loginfo(f"  当前网格总数: {animal_info['count']}")
            rospy.loginfo(f"  任务累计总数: {self.mission_total_count}")
            rospy.loginfo(f"  累计动物分布: {self.mission_total_animals}")
            rospy.loginfo(f"  是否首次访问: {should_accumulate}")
            if should_accumulate:
                rospy.loginfo(f"  数据已持久化保存")
    
    def reset_mission_totals(self):
        """重置任务累计统计"""
        self.mission_total_animals = {
            "Wolf": 0,
            "Elephant": 0,
            "Peacock": 0,
            "Monkey": 0,
            "Tiger": 0
        }
        self.mission_total_count = 0
        self.save_mission_totals()
        rospy.loginfo("�� 任务累计统计已重置")
    
    def reset_visited_positions(self):
        """重置已访问位置记录（用于新任务开始）"""
        self.visited_animal_positions.clear()
        self.mission_started = False
        self.mission_completed = False
        rospy.loginfo("�� 已访问位置记录已重置，准备开始新任务")
    
    def reset_publish_cooldown(self):
        """重置发布冷却时间"""
        self.last_detection_time.clear()
        rospy.loginfo("�� 发布冷却时间已重置")
    
    def debug_coordinate_conversion(self, x, y):
        """调试坐标转换"""
        b_index = int(x / self.grid_size)
        a_index = int(y / self.grid_size)
        grid_pos = f"A{9-a_index}B{b_index+1}"
        
        rospy.loginfo(f"�� 坐标转换调试:")
        rospy.loginfo(f"  输入坐标: ({x:.2f}, {y:.2f})")
        rospy.loginfo(f"  网格大小: {self.grid_size}")
        rospy.loginfo(f"  B索引: {b_index} (对应B{b_index+1})")
        rospy.loginfo(f"  A索引: {a_index} (对应A{9-a_index})")
        rospy.loginfo(f"  结果网格: {grid_pos}")
        
        return grid_pos
    
    def run(self):
        """运行节点"""
        rospy.loginfo("动物检测发布器（YOLO集成版）开始运行...")
        rospy.loginfo("订阅话题:")
        rospy.loginfo("  - /mavros/local_position/odom (无人机位置)")
        rospy.loginfo("  - /planned_path (任务目标航点)")
        rospy.loginfo("  - /bboxes_pub (YOLO识别结果)")
        rospy.loginfo("发布话题: /grid_animal_report (String消息类型)")
        rospy.loginfo("消息格式: JSON字符串 - 与现有YOLO系统兼容")
        rospy.loginfo("")
        rospy.loginfo("工作模式:")
        rospy.loginfo("1. �� 自动模式: 从任务目标航点自动获取网格位置并发布动物信息")
        rospy.loginfo("2. �� YOLO集成: 结合YOLO识别结果和预设动物数据")
        rospy.loginfo("3. �� 手动模式: 直接发布指定网格: publisher.publish_target_grid_animal('A6B2')")
        rospy.loginfo("4. �� 位置模式: 等待位置变化自动检测（如果启用）")
        rospy.loginfo("")
        rospy.loginfo("支持所有预设动物位置: A3B7, A5B6, A4B5, A3B5, A8B2, A2B2, A6B2")
        rospy.loginfo("")
        rospy.loginfo("�� 功能说明:")
        rospy.loginfo("- total_animals字段显示任务累计总量")
        rospy.loginfo("- 防重复机制: 3秒内不重复发布相同网格")
        rospy.loginfo("- YOLO集成: 优先使用YOLO识别结果(置信度≥0.9)，超时后使用预设数据")
        rospy.loginfo("- 累加功能: YOLO检测结果会累加到累计统计中")
        rospy.loginfo("- 可以使用 'rostopic echo /grid_animal_report' 查看消息")
        rospy.loginfo("- 重置累计: publisher.reset_mission_totals()")
        rospy.loginfo("- 重置冷却: publisher.reset_publish_cooldown()")
        rospy.loginfo("")
        
        # 保持节点运行
        rospy.spin()

def main():
    try:
        publisher = AnimalDetectionPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("动物检测发布器已停止")

if __name__ == '__main__':
    main() 
