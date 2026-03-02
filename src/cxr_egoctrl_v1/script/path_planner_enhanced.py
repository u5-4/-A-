#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from geometry_msgs.msg import PoseArray, Pose, Point
from std_msgs.msg import Header, String
import math
import time

class GreedyPathPlannerComplete:
    def __init__(self):
        rospy.init_node('greedy_path_planner_complete', anonymous=True)
        
        # 发布者：发布规划好的路径点（遍历路径，地面站显示用）
        self.path_pub = rospy.Publisher('/planned_path', PoseArray, queue_size=1)
        
        # 发布者：发布完整路径（遍历+返航，主控制器用）
        self.complete_path_pub = rospy.Publisher('/complete_path', PoseArray, queue_size=1)
        
        # 发布者：发布回到起飞点的路径（兼容性）
        self.return_path_pub = rospy.Publisher('/return_path', PoseArray, queue_size=1)
        
        # 发布者：发布规划统计信息
        self.stats_pub = rospy.Publisher('/path_planning_stats', String, queue_size=1)
        
        # 订阅者：订阅禁飞区域信息
        self.no_fly_sub = rospy.Subscriber('/no_fly_zones_legacy', PoseArray, self.no_fly_zone_callback)
        
        # 参数
        self.takeoff_height = rospy.get_param('~takeoff_height', 1.0)
        self.grid_size = rospy.get_param('~grid_size', 0.5)
        self.grid_width = rospy.get_param('~grid_width', 9)
        self.grid_height = rospy.get_param('~grid_height', 7)
        
        # 贪心算法参数
        self.min_distance_threshold = rospy.get_param('~min_distance_threshold', 0.3)  # 最小航点间距
        self.path_optimization = rospy.get_param('~path_optimization', True)           # 路径优化
        self.use_nearest_neighbor = rospy.get_param('~use_nearest_neighbor', False)   # 使用蛇形遍历
        
        # 初始化网格坐标点
        self.all_grid_points = self.initialize_grid_points()
        
        # 禁飞区域
        self.no_fly_zones = []
        self.no_fly_zone_grid = set()  # 网格坐标形式的禁飞区域
        
        # 规划好的路径
        self.planned_path = []  # 遍历路径（地面站显示用）
        self.return_path = []   # 返航路径
        self.complete_path = [] # 完整路径（遍历+返航，主控制器用）
        
        # 路径发布控制
        self.path_published = False  # 标记是否已发布路径
        self.last_path_hash = None   # 记录上次路径的哈希值
        
        rospy.loginfo("完整版贪心算法路径规划器已初始化（分离遍历路径和返航路径）")
        rospy.loginfo(f"最小航点间距: {self.min_distance_threshold}m")
        rospy.loginfo(f"路径优化: {self.path_optimization}")
        rospy.loginfo(f"使用蛇形遍历: {not self.use_nearest_neighbor}")
        
    def initialize_grid_points(self):
        """初始化所有网格坐标点"""
        points = []
        point_id = 0
        
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                # 新坐标系：X轴平行于B列（向前），Y轴平行于A列（向左）
                # X轴：B1到B7（向前），Y轴：A9到A1（向左）
                # A9B1为原点(0,0)，A索引减小时Y增加，B索引增加时X增加
                x = row * self.grid_size                          # X轴：B1到B7（向前）
                y = (self.grid_width - 1 - col) * self.grid_size  # Y轴：A9到A1（向左）
                z = self.takeoff_height
                
                point = {
                    'id': point_id,
                    'grid_pos': f"A{col+1}B{row+1}",
                    'coordinate': [x, y, z],
                    'is_accessible': True,
                    'row': row,
                    'col': col,
                    'visited': False  # 标记是否已访问
                }
                points.append(point)
                point_id += 1
                
        rospy.loginfo(f"初始化了 {len(points)} 个网格点")
        return points
    
    def no_fly_zone_callback(self, msg):
        """处理禁飞区域消息"""
        start_time = time.time()
        rospy.loginfo(f"收到禁飞区域消息，包含 {len(msg.poses)} 个点")
        
        self.no_fly_zones = []
        self.no_fly_zone_grid = set()
        
        for pose in msg.poses:
            z_coord = pose.position.z if abs(pose.position.z) > 0.001 else self.takeoff_height
            
            no_fly_point = {
                'x': pose.position.x,
                'y': pose.position.y,
                'z': z_coord
            }
            self.no_fly_zones.append(no_fly_point)
            
            # 转换为网格坐标并添加到禁飞区域集合
            grid_col, grid_row = self.world_to_grid_coordinates(pose.position.x, pose.position.y)
            self.no_fly_zone_grid.add((grid_col, grid_row))
            
            rospy.loginfo(f"禁飞区域点: ({pose.position.x:.1f}, {pose.position.y:.1f}, {z_coord:.1f}) -> 网格({grid_col}, {grid_row})")
        
        # 检测连续禁飞区域
        self.detect_continuous_no_fly_zones()
        
        self.update_grid_accessibility()
        self.plan_complete_path()
        
        # 立即发布路径
        self.publish_all_paths()
        
        # 计算并记录耗时
        end_time = time.time()
        planning_time = end_time - start_time
        rospy.loginfo(f"完整版路径规划耗时: {planning_time:.3f}秒")
        
        # 发布规划耗时统计
        self.publish_planning_statistics(planning_time, len(msg.poses))
    
    def world_to_grid_coordinates(self, x, y):
        """世界坐标转换为网格坐标"""
        col = int(y / self.grid_size)
        row = int(x / self.grid_size)
        return col, row
    
    def detect_continuous_no_fly_zones(self):
        """检测连续禁飞区域"""
        continuous_zones = []
        
        # 检测水平连续区域（与X轴平行）
        for row in range(self.grid_height):
            continuous_cols = []
            for col in range(self.grid_width):
                if (col, row) in self.no_fly_zone_grid:
                    continuous_cols.append(col)
                else:
                    if len(continuous_cols) >= 2:  # 至少2个连续点
                        continuous_zones.append(('horizontal', row, continuous_cols))
                    continuous_cols = []
            
            # 检查行末的连续区域
            if len(continuous_cols) >= 2:
                continuous_zones.append(('horizontal', row, continuous_cols))
        
        # 检测垂直连续区域（与Y轴平行）
        for col in range(self.grid_width):
            continuous_rows = []
            for row in range(self.grid_height):
                if (col, row) in self.no_fly_zone_grid:
                    continuous_rows.append(row)
                else:
                    if len(continuous_rows) >= 2:  # 至少2个连续点
                        continuous_zones.append(('vertical', col, continuous_rows))
                    continuous_rows = []
            
            # 检查列末的连续区域
            if len(continuous_rows) >= 2:
                continuous_zones.append(('vertical', col, continuous_rows))
        
        rospy.loginfo(f"检测到 {len(continuous_zones)} 个连续禁飞区域:")
        for zone_type, fixed_coord, continuous_coords in continuous_zones:
            rospy.loginfo(f"  {zone_type}连续区域: {zone_type=='horizontal' and 'B'+str(fixed_coord+1) or 'A'+str(fixed_coord+1)}, 连续坐标: {continuous_coords}")
    
    def update_grid_accessibility(self):
        """根据禁飞区域更新网格点的可访问性"""
        for point in self.all_grid_points:
            point['is_accessible'] = True
            point['visited'] = False
        
        for no_fly_zone in self.no_fly_zones:
            for point in self.all_grid_points:
                xy_distance = math.sqrt(
                    (point['coordinate'][0] - no_fly_zone['x'])**2 +
                    (point['coordinate'][1] - no_fly_zone['y'])**2
                )
                
                if xy_distance < self.grid_size * 0.7:
                    point['is_accessible'] = False
                    rospy.logwarn(f"网格点 {point['grid_pos']} 被标记为禁飞区域")
                    break
    
    def plan_complete_path(self):
        """规划完整路径（确保遍历所有可访问点）"""
        accessible_points = [point for point in self.all_grid_points if point['is_accessible']]
        
        if not accessible_points:
            rospy.logwarn("没有可访问的网格点！")
            self.planned_path = []
            self.return_path = []
            self.complete_path = []
            return
        
        rospy.loginfo(f"开始完整版路径规划，可访问点数量: {len(accessible_points)}")
        
        # 使用改进的贪心算法（分离遍历路径和回程路径）
        self.planned_path, self.return_path = self.improved_greedy_path_planning(accessible_points)
        
        # 路径优化（保留所有必要的航点）
        if self.path_optimization and len(self.planned_path) > 2:
            self.planned_path = self.optimize_path_keep_all_points(self.planned_path)
        
        # 生成完整路径：遍历路径 + 返航路径
        self.complete_path = self.planned_path.copy()
        if self.return_path:
            self.complete_path.extend(self.return_path)
        
        # 验证路径安全性
        self.verify_path_safety()
        
        # 统计信息
        total_points = len(self.all_grid_points)
        no_fly_points = total_points - len(accessible_points)
        coverage = (len(accessible_points) / total_points) * 100
        traversal_length = self.calculate_path_length(self.planned_path)
        return_length = self.calculate_path_length(self.return_path)
        total_length = traversal_length + return_length
        
        rospy.loginfo(f"完整版路径规划完成")
        rospy.loginfo(f"遍历路径: {len(self.planned_path)} 个点, 长度: {traversal_length:.2f}m")
        rospy.loginfo(f"返航路径: {len(self.return_path)} 个点, 长度: {return_length:.2f}m")
        rospy.loginfo(f"完整路径: {len(self.complete_path)} 个点, 总长度: {total_length:.2f}m")
        rospy.loginfo(f"网格统计: 总点数={total_points}, 可访问={len(accessible_points)}, 禁飞={no_fly_points}, 覆盖率={coverage:.1f}%")
        
        # 验证是否遍历了所有可访问点
        self.verify_complete_coverage(accessible_points)
        
        # 打印路径概览
        rospy.loginfo("遍历路径概览:")
        for i in range(0, len(self.planned_path), 10):
            point = self.planned_path[i]
            rospy.loginfo(f"  遍历航点{i+1}: {point['grid_pos']} ({point['coordinate'][0]:.1f}, {point['coordinate'][1]:.1f})")
        
        if self.return_path:
            rospy.loginfo("返航路径概览:")
            for i in range(0, len(self.return_path), 5):
                point = self.return_path[i]
                rospy.loginfo(f"  返航航点{i+1}: {point['grid_pos']} ({point['coordinate'][0]:.1f}, {point['coordinate'][1]:.1f})")
    
    def improved_greedy_path_planning(self, accessible_points):
        """改进的贪心路径规划算法（分离遍历路径和回程路径）"""
        # 找到起飞点A9B1
        start_point = None
        for point in accessible_points:
            if point['grid_pos'] == 'A9B1':
                start_point = point
                break
        
        if not start_point:
            rospy.logwarn("未找到起飞点A9B1，使用第一个可访问点")
            start_point = accessible_points[0]
        
        # 遍历路径（不包含回到起飞点的部分）
        traversal_path = [start_point]
        visited = set([start_point['grid_pos']])
        current = start_point
        
        # 生成所有需要访问的网格坐标（排除禁飞区和起点）
        cells_to_visit = []
        for point in accessible_points:
            if point['grid_pos'] != start_point['grid_pos']:
                cells_to_visit.append(point)
        
        rospy.loginfo(f"开始贪心算法，起点: {start_point['grid_pos']}, 待访问点: {len(cells_to_visit)}")
        
        # 贪心算法：每次选择最近的未访问网格
        while cells_to_visit:
            # 计算当前点到所有未访问点的距离
            nearest = min(
                cells_to_visit,
                key=lambda x: self.calculate_distance(current, x)
            )
            
            # 检查路径安全性
            if self.is_safe_path(current, nearest):
                # 直接连接
                traversal_path.append(nearest)
                rospy.loginfo(f"直接连接: {current['grid_pos']} -> {nearest['grid_pos']}")
            else:
                # 需要绕行
                detour_path = self.find_safe_detour_path(current, nearest)
                if detour_path:
                    traversal_path.extend(detour_path[1:])  # 跳过起点（避免重复）
                    rospy.loginfo(f"绕行路径: {current['grid_pos']} -> {[p['grid_pos'] for p in detour_path[1:]]} -> {nearest['grid_pos']}")
                else:
                    rospy.logwarn(f"无法找到从 {current['grid_pos']} 到 {nearest['grid_pos']} 的安全路径")
                    # 强制连接（不推荐）
                    traversal_path.append(nearest)
            
            # 更新状态
            visited.add(nearest['grid_pos'])
            cells_to_visit.remove(nearest)
            current = nearest
        
        rospy.loginfo(f"贪心算法完成，遍历路径长度: {len(traversal_path)}")
        
        # 寻找回到起飞点的路径（独立处理）
        return_path = self.find_return_path_to_start(current, start_point)
        if return_path:
            rospy.loginfo(f"找到回到起飞点路径: {[p['grid_pos'] for p in return_path]}")
        else:
            rospy.logwarn("无法找到回到起飞点的安全路径")
            return_path = []
        
        return traversal_path, return_path
    
    def is_safe_path(self, start_point, target_point):
        """检查两点间路径是否安全"""
        start_x, start_y = start_point['coordinate'][0], start_point['coordinate'][1]
        target_x, target_y = target_point['coordinate'][0], target_point['coordinate'][1]
        
        # 计算路径上的所有中间点
        dx = target_x - start_x
        dy = target_y - start_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 0.1:  # 距离太近，不需要检查
            return True
            
        # 检查路径上的多个点
        num_checks = max(5, int(distance / (self.grid_size * 0.2)))
        for i in range(1, num_checks):
            t = i / num_checks
            mid_x = start_x + t * dx
            mid_y = start_y + t * dy
            
            # 检查中间点是否在禁飞区域内
            for no_fly_zone in self.no_fly_zones:
                wall_distance = math.sqrt((mid_x - no_fly_zone['x'])**2 + (mid_y - no_fly_zone['y'])**2)
                if wall_distance < self.grid_size * 0.8:  # 更严格的检测
                    return False  # 不安全
        
        return True  # 安全
    
    def find_safe_detour_path(self, start_point, target_point):
        """寻找安全的绕行路径"""
        start_x, start_y = start_point['coordinate'][0], start_point['coordinate'][1]
        target_x, target_y = target_point['coordinate'][0], target_point['coordinate'][1]
        
        # 寻找所有可能的绕行路径
        detour_paths = []
        
        # 策略1: 向上绕行（垂直绕行 - 优先）
        up_detour = self.try_up_detour_strategy(start_point, target_point)
        if up_detour:
            detour_paths.append(("垂直向上", up_detour))
        
        # 策略2: 向下绕行（垂直绕行 - 优先）
        down_detour = self.try_down_detour_strategy(start_point, target_point)
        if down_detour:
            detour_paths.append(("垂直向下", down_detour))
        
        # 策略3: 左右绕行（水平绕行 - 备选）
        left_detour = self.try_left_detour_strategy(start_point, target_point)
        if left_detour:
            detour_paths.append(("水平向左", left_detour))
        
        right_detour = self.try_right_detour_strategy(start_point, target_point)
        if right_detour:
            detour_paths.append(("水平向右", right_detour))
        
        # 优先选择垂直绕行路径
        vertical_detours = [path for name, path in detour_paths if "垂直" in name]
        if vertical_detours:
            # 在垂直绕行中选择最短的
            best_vertical = min(vertical_detours, key=lambda path: self.calculate_path_length(path))
            rospy.loginfo(f"选择垂直绕行路径，长度: {self.calculate_path_length(best_vertical):.2f}m")
            return best_vertical
        
        # 如果没有垂直绕行，选择水平绕行
        if detour_paths:
            best_path = min(detour_paths, key=lambda item: self.calculate_path_length(item[1]))
            rospy.loginfo(f"选择水平绕行路径: {best_path[0]}，长度: {self.calculate_path_length(best_path[1]):.2f}m")
            return best_path[1]
        
        rospy.logwarn(f"无法找到绕行路径: ({start_x:.1f},{start_y:.1f}) -> ({target_x:.1f},{target_y:.1f})")
        return None
    
    def find_return_path_to_start(self, current_point, start_point):
        """使用纯粹贪心算法找到回到起飞点的最短路径"""
        rospy.loginfo(f"开始寻找回到起飞点 {start_point['grid_pos']} 的路径")
        
        # 如果已经在起飞点附近，直接返回
        if self.calculate_distance(current_point, start_point) < self.grid_size * 0.5:
            rospy.loginfo("已在起飞点附近，无需额外路径")
            return []
        
        # 使用A*算法寻找最短路径
        return self.a_star_pathfinding(current_point, start_point)
    
    def a_star_pathfinding(self, start_point, target_point):
        """A*算法寻找最短路径（避开禁飞区域）"""
        rospy.loginfo(f"A*算法: {start_point['grid_pos']} -> {target_point['grid_pos']}")
        
        # 初始化开放列表和关闭列表
        open_list = [(0, start_point)]  # (f_score, point)
        closed_set = set()
        came_from = {}
        g_score = {start_point['grid_pos']: 0}  # 从起点到当前点的实际代价
        f_score = {start_point['grid_pos']: self.calculate_distance(start_point, target_point)}  # 估计总代价
        
        while open_list:
            # 获取f_score最小的点
            current_f, current_point = open_list.pop(0)
            current_pos = current_point['grid_pos']
            
            # 如果到达目标点
            if current_pos == target_point['grid_pos']:
                return self.reconstruct_path(came_from, current_point)
            
            closed_set.add(current_pos)
            
            # 获取相邻点
            neighbors = self.get_safe_neighbors(current_point)
            
            for neighbor in neighbors:
                neighbor_pos = neighbor['grid_pos']
                
                if neighbor_pos in closed_set:
                    continue
                
                # 计算从起点经过当前点到邻居的代价
                tentative_g_score = g_score[current_pos] + self.calculate_distance(current_point, neighbor)
                
                # 如果找到更好的路径
                if neighbor_pos not in g_score or tentative_g_score < g_score[neighbor_pos]:
                    came_from[neighbor_pos] = current_point
                    g_score[neighbor_pos] = tentative_g_score
                    f_score[neighbor_pos] = tentative_g_score + self.calculate_distance(neighbor, target_point)
                    
                    # 将邻居添加到开放列表
                    if neighbor_pos not in [pos for _, point in open_list for pos in [point['grid_pos']]]:
                        open_list.append((f_score[neighbor_pos], neighbor))
            
            # 按f_score排序开放列表
            open_list.sort(key=lambda x: x[0])
        
        rospy.logwarn(f"A*算法无法找到从 {start_point['grid_pos']} 到 {target_point['grid_pos']} 的路径")
        return None
    
    def get_safe_neighbors(self, point):
        """获取安全的相邻点"""
        neighbors = []
        x, y = point['coordinate'][0], point['coordinate'][1]
        
        # 四个方向的相邻点
        directions = [
            (0, self.grid_size),   # 上
            (0, -self.grid_size),  # 下
            (-self.grid_size, 0),  # 左
            (self.grid_size, 0)    # 右
        ]
        
        for dx, dy in directions:
            neighbor_x = x + dx
            neighbor_y = y + dy
            
            # 检查边界
            if (0 <= neighbor_x <= (self.grid_height - 1) * self.grid_size and 
                0 <= neighbor_y <= (self.grid_width - 1) * self.grid_size):
                
                neighbor = self.find_grid_point(neighbor_x, neighbor_y)
                if neighbor and neighbor['is_accessible']:
                    # 检查路径安全性
                    if self.is_safe_path(point, neighbor):
                        neighbors.append(neighbor)
        
        return neighbors
    
    def reconstruct_path(self, came_from, current_point):
        """重建路径"""
        path = []
        current_pos = current_point['grid_pos']
        
        while current_pos in came_from:
            path.append(current_point)
            current_point = came_from[current_pos]
            current_pos = current_point['grid_pos']
        
        path.append(current_point)
        path.reverse()
        
        rospy.loginfo(f"A*路径重建完成，长度: {len(path)}")
        return path
    
    def try_up_detour_strategy(self, start_point, target_point):
        """尝试向上绕行策略（真正的垂直绕行）"""
        start_x, start_y = start_point['coordinate'][0], start_point['coordinate'][1]
        target_x, target_y = target_point['coordinate'][0], target_point['coordinate'][1]
        
        # 向上移动一行（垂直绕行）
        up_y = start_y + self.grid_size
        if up_y <= 3.0:  # 确保不超出网格范围
            # 检查向上路径是否安全（垂直移动）
            up_point = self.find_grid_point(start_x, up_y)
            if up_point and up_point['is_accessible'] and self.is_safe_path(start_point, up_point):
                # 检查水平移动是否安全（在更高层）
                mid_point = self.find_grid_point(target_x, up_y)
                if mid_point and mid_point['is_accessible'] and self.is_safe_path(up_point, mid_point):
                    # 检查向下移动是否安全（垂直下降）
                    if self.is_safe_path(mid_point, target_point):
                        rospy.loginfo(f"使用向上绕行: ({start_x:.1f},{start_y:.1f}) -> 上({start_x:.1f},{up_y:.1f}) -> 平({target_x:.1f},{up_y:.1f}) -> 下({target_x:.1f},{target_y:.1f})")
                        return [start_point, up_point, mid_point, target_point]
        
        return None
    
    def try_down_detour_strategy(self, start_point, target_point):
        """尝试向下绕行策略（真正的垂直绕行）"""
        start_x, start_y = start_point['coordinate'][0], start_point['coordinate'][1]
        target_x, target_y = target_point['coordinate'][0], target_point['coordinate'][1]
        
        # 向下移动一行（垂直绕行）
        down_y = start_y - self.grid_size
        if down_y >= 0.0:  # 确保不超出网格范围
            # 检查向下路径是否安全（垂直移动）
            down_point = self.find_grid_point(start_x, down_y)
            if down_point and down_point['is_accessible'] and self.is_safe_path(start_point, down_point):
                # 检查水平移动是否安全（在更低层）
                mid_point = self.find_grid_point(target_x, down_y)
                if mid_point and mid_point['is_accessible'] and self.is_safe_path(down_point, mid_point):
                    # 检查向上移动是否安全（垂直上升）
                    if self.is_safe_path(mid_point, target_point):
                        rospy.loginfo(f"使用向下绕行: ({start_x:.1f},{start_y:.1f}) -> 下({start_x:.1f},{down_y:.1f}) -> 平({target_x:.1f},{down_y:.1f}) -> 上({target_x:.1f},{target_y:.1f})")
                        return [start_point, down_point, mid_point, target_point]
        
        return None
    
    def try_left_detour_strategy(self, start_point, target_point):
        """尝试向左绕行策略（水平绕行）"""
        start_x, start_y = start_point['coordinate'][0], start_point['coordinate'][1]
        target_x, target_y = target_point['coordinate'][0], target_point['coordinate'][1]
        
        # 向左移动一列（水平绕行）
        left_x = start_x - self.grid_size
        if left_x >= 0.0:  # 确保不超出网格范围
            # 检查向左路径是否安全（水平移动）
            left_point = self.find_grid_point(left_x, start_y)
            if left_point and left_point['is_accessible'] and self.is_safe_path(start_point, left_point):
                # 检查垂直移动是否安全（在更左层）
                mid_point = self.find_grid_point(left_x, target_y)
                if mid_point and mid_point['is_accessible'] and self.is_safe_path(left_point, mid_point):
                    # 检查向右移动是否安全（水平移动）
                    if self.is_safe_path(mid_point, target_point):
                        rospy.loginfo(f"使用向左绕行: ({start_x:.1f},{start_y:.1f}) -> 左({left_x:.1f},{start_y:.1f}) -> 平({left_x:.1f},{target_y:.1f}) -> 右({target_x:.1f},{target_y:.1f})")
                        return [start_point, left_point, mid_point, target_point]
        
        return None
    
    def try_right_detour_strategy(self, start_point, target_point):
        """尝试向右绕行策略（水平绕行）"""
        start_x, start_y = start_point['coordinate'][0], start_point['coordinate'][1]
        target_x, target_y = target_point['coordinate'][0], target_point['coordinate'][1]
        
        # 向右移动一列（水平绕行）
        right_x = start_x + self.grid_size
        if right_x <= 4.0:  # 确保不超出网格范围
            # 检查向右路径是否安全（水平移动）
            right_point = self.find_grid_point(right_x, start_y)
            if right_point and right_point['is_accessible'] and self.is_safe_path(start_point, right_point):
                # 检查垂直移动是否安全（在更右层）
                mid_point = self.find_grid_point(right_x, target_y)
                if mid_point and mid_point['is_accessible'] and self.is_safe_path(right_point, mid_point):
                    # 检查向左移动是否安全（水平移动）
                    if self.is_safe_path(mid_point, target_point):
                        rospy.loginfo(f"使用向右绕行: ({start_x:.1f},{start_y:.1f}) -> 右({right_x:.1f},{start_y:.1f}) -> 平({right_x:.1f},{target_y:.1f}) -> 左({target_x:.1f},{target_y:.1f})")
                        return [start_point, right_point, mid_point, target_point]
        
        return None
    
    def find_grid_point(self, x, y):
        """根据坐标找到网格点"""
        for point in self.all_grid_points:
            if abs(point['coordinate'][0] - x) < 0.1 and abs(point['coordinate'][1] - y) < 0.1:
                return point
        return None
    
    def verify_path_safety(self):
        """验证整个路径的安全性"""
        if len(self.planned_path) < 2:
            return
        
        unsafe_segments = []
        for i in range(len(self.planned_path) - 1):
            start_point = self.planned_path[i]
            target_point = self.planned_path[i + 1]
            
            if not self.is_safe_path(start_point, target_point):
                unsafe_segments.append((start_point['grid_pos'], target_point['grid_pos']))
        
        if unsafe_segments:
            rospy.logwarn(f"发现 {len(unsafe_segments)} 个不安全路径段:")
            for start, target in unsafe_segments:
                rospy.logwarn(f"  {start} -> {target}")
        else:
            rospy.loginfo("✅ 所有路径段都通过安全检查")
    
    def optimize_path_keep_all_points(self, path):
        """优化路径但保留所有必要的航点"""
        if len(path) < 3:
            return path
        
        optimized_path = [path[0]]  # 保留起点
        
        for i in range(1, len(path)):
            current_point = path[i]
            last_point = optimized_path[-1]
            
            # 计算距离
            distance = self.calculate_distance(last_point, current_point)
            
            # 如果距离太近且不是绕行点，跳过这个点
            if distance < self.min_distance_threshold and not self.is_detour_point(current_point):
                continue
            
            optimized_path.append(current_point)
        
        rospy.loginfo(f"路径优化: {len(path)} -> {len(optimized_path)} 个航点")
        return optimized_path
    
    def is_detour_point(self, point):
        """判断是否为绕行点"""
        # 检查是否在绕行路径上
        for no_fly_zone in self.no_fly_zones:
            distance = math.sqrt(
                (point['coordinate'][0] - no_fly_zone['x'])**2 +
                (point['coordinate'][1] - no_fly_zone['y'])**2
            )
            if distance < self.grid_size * 1.5:  # 绕行点通常在禁飞区附近
                return True
        return False
    
    def verify_complete_coverage(self, accessible_points):
        """验证是否遍历了所有可访问点"""
        visited_positions = set()
        for point in self.planned_path:
            visited_positions.add(point['grid_pos'])
        
        accessible_positions = set(point['grid_pos'] for point in accessible_points)
        missing_positions = accessible_positions - visited_positions
        
        if missing_positions:
            rospy.logwarn(f"未遍历的点: {missing_positions}")
        else:
            rospy.loginfo(f"✅ 已遍历所有 {len(accessible_points)} 个可访问点")
    
    def calculate_distance(self, point1, point2):
        """计算两点间距离"""
        return math.sqrt(
            (point1['coordinate'][0] - point2['coordinate'][0])**2 +
            (point1['coordinate'][1] - point2['coordinate'][1])**2
        )
    
    def calculate_path_length(self, path=None):
        """计算路径总长度"""
        if path is None:
            path = self.planned_path
            
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path) - 1):
            total_length += self.calculate_distance(path[i], path[i + 1])
        
        return total_length
    
    def publish_all_paths(self):
        """发布所有路径到不同话题"""
        # 1. 发布遍历路径到 /planned_path（地面站显示用）
        self.publish_traversal_path()
        
        # 2. 发布完整路径到 /complete_path（主控制器用）
        self.publish_complete_path()
        
        # 3. 发布返航路径到 /return_path（兼容性）
        self.publish_return_path()
    
    def publish_traversal_path(self):
        """发布遍历路径到 /planned_path（地面站显示用）"""
        if not self.planned_path:
            rospy.logwarn("没有遍历路径，跳过发布")
            return

        pose_array = PoseArray()
        pose_array.header = Header()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = "map"
        
        for point in self.planned_path:
            pose = Pose()
            pose.position.x = point['coordinate'][0]
            pose.position.y = point['coordinate'][1]
            pose.position.z = point['coordinate'][2]
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)
        
        self.path_pub.publish(pose_array)
        rospy.loginfo(f"✅ 发布遍历路径到 /planned_path: {len(self.planned_path)} 个航点（地面站显示用）")
    
    def publish_complete_path(self):
        """发布完整路径到 /complete_path（主控制器用）"""
        if not self.complete_path:
            rospy.logwarn("没有完整路径，跳过发布")
            return

        pose_array = PoseArray()
        pose_array.header = Header()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = "map"
        
        for point in self.complete_path:
            pose = Pose()
            pose.position.x = point['coordinate'][0]
            pose.position.y = point['coordinate'][1]
            pose.position.z = point['coordinate'][2]
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)
        
        self.complete_path_pub.publish(pose_array)
        rospy.loginfo(f"✅ 发布完整路径到 /complete_path: {len(self.complete_path)} 个航点（主控制器用）")
        rospy.loginfo(f"  包含: {len(self.planned_path)} 个遍历航点 + {len(self.return_path)} 个返航航点")
    
    def publish_return_path(self):
        """发布返航路径到 /return_path（兼容性）"""
        if not self.return_path:
            rospy.loginfo("没有返航路径，跳过发布")
            return

        pose_array = PoseArray()
        pose_array.header = Header()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = "map"
        
        for point in self.return_path:
            pose = Pose()
            pose.position.x = point['coordinate'][0]
            pose.position.y = point['coordinate'][1]
            pose.position.z = point['coordinate'][2]
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)
        
        self.return_path_pub.publish(pose_array)
        rospy.loginfo(f"✅ 发布返航路径到 /return_path: {len(self.return_path)} 个航点（兼容性）")
    
    def publish_planning_statistics(self, planning_time, no_fly_zone_count):
        """发布路径规划统计信息"""
        traversal_length = self.calculate_path_length(self.planned_path)
        return_length = self.calculate_path_length(self.return_path)
        total_length = traversal_length + return_length
        
        stats_msg = f"完整版规划耗时:{planning_time:.3f}s, 禁飞区数量:{no_fly_zone_count}, 遍历路径点数:{len(self.planned_path)}, 返航路径点数:{len(self.return_path)}, 总路径长度:{total_length:.2f}m"
        self.stats_pub.publish(stats_msg)
        rospy.loginfo(f"发布完整版规划统计: {stats_msg}")

def main():
    try:
        planner = GreedyPathPlannerComplete()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("完整版贪心算法路径规划器已停止")

if __name__ == '__main__':
    main()
