#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header

class CoordinateConverter:
    def __init__(self):
        rospy.init_node('coordinate_converter', anonymous=True)
        
        # 订阅原始禁飞区域
        self.raw_sub = rospy.Subscriber('/no_fly_zones_raw', PoseArray, self.raw_callback)
        
        # 发布转换后的禁飞区域
        self.converted_pub = rospy.Publisher('/no_fly_zones_legacy', PoseArray, queue_size=1)
        
        # 坐标转换参数 - 修正版本
        # 地面站坐标：A1-A9 (0-400cm), B1-B7 (0-300cm)
        # 路径规划器坐标：X (0-4.0m), Y (0-3.0m)
        self.scale_x = 0.01  # 400cm -> 4.0m (400 * 0.01 = 4.0)
        self.scale_y = 0.01  # 300cm -> 3.0m (300 * 0.01 = 3.0)
        self.offset_x = 0.0
        self.offset_y = 0.0
        
        rospy.loginfo("=== 坐标转换器已启动 ===")
        rospy.loginfo(f"X轴缩放: {self.scale_x} (400cm -> 4.0m)")
        rospy.loginfo(f"Y轴缩放: {self.scale_y} (300cm -> 3.0m)")
        rospy.loginfo("坐标范围: A1-A9 (0-400cm), B1-B7 (0-300cm)")
        
    def raw_callback(self, msg):
        """处理原始禁飞区域数据"""
        converted_msg = PoseArray()
        converted_msg.header = Header()
        converted_msg.header.stamp = rospy.Time.now()
        converted_msg.header.frame_id = "map"
        
        for pose in msg.poses:
            # 坐标转换 (cm -> m)
            converted_x = pose.position.x * self.scale_x + self.offset_x
            converted_y = pose.position.y * self.scale_y + self.offset_y
            converted_z = 1.0  # 固定高度为1.0米
            
            # 检查是否在有效范围内
            if 0.0 <= converted_x <= 4.0 and 0.0 <= converted_y <= 3.0:
                converted_pose = Pose()
                converted_pose.position.x = converted_x
                converted_pose.position.y = converted_y
                converted_pose.position.z = converted_z
                converted_pose.orientation.w = 1.0
                converted_msg.poses.append(converted_pose)
                
                rospy.loginfo(f"✅ 坐标转换: ({pose.position.x:.1f}cm, {pose.position.y:.1f}cm) -> ({converted_x:.2f}m, {converted_y:.2f}m)")
            else:
                rospy.logwarn(f"❌ 坐标超出范围，跳过: ({pose.position.x:.1f}cm, {pose.position.y:.1f}cm) -> ({converted_x:.2f}m, {converted_y:.2f}m)")
                rospy.logwarn(f"   有效范围: X(0-400cm), Y(0-300cm)")
        
        if converted_msg.poses:
            self.converted_pub.publish(converted_msg)
            rospy.loginfo(f"✅ 发布转换后的禁飞区域: {len(converted_msg.poses)} 个点")
        else:
            rospy.logwarn("❌ 没有有效的转换坐标")
            rospy.logwarn("请检查地面站发送的坐标是否在有效范围内")

def main():
    try:
        converter = CoordinateConverter()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("坐标转换器已停止")

if __name__ == '__main__':
    main() 