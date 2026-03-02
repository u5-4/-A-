#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import time
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Bool

class AutoPathPublisher:
    def __init__(self):
        rospy.init_node('auto_path_publisher', anonymous=True)
        
        # 订阅者
        rospy.Subscriber('/path_ready', Bool, self.path_ready_callback)
        rospy.Subscriber('/planned_path', PoseArray, self.planned_path_callback)
        
        # 状态变量
        self.path_ready_received = False
        self.path_published = False
        self.last_path_count = 0
        
        print("=== 自动路径发布器 ===")
        print("监控路径就绪信号并自动发布路径")
        print("-" * 50)
        
    def path_ready_callback(self, msg):
        """路径就绪回调"""
        if msg.data and not self.path_published:
            self.path_ready_received = True
            print(f"✅ 收到路径就绪信号: {msg.data}")
            print("等待路径规划器发布路径...")
            
            # 等待一段时间让路径规划器发布路径
            rospy.sleep(1.0)
            
            # 如果还没有收到路径，发送任务启动信号
            if not self.path_published:
                print("�� 自动发送任务启动信号...")
                self.send_mission_start_signal()
                
    def planned_path_callback(self, msg):
        """规划路径回调"""
        if len(msg.poses) > 0:
            self.path_published = True
            path_count = len(msg.poses)
            
            if path_count != self.last_path_count:
                print(f"�� 收到规划路径: {path_count} 个航点")
                print(f"   第一个航点: X={msg.poses[0].position.x:.2f}, Y={msg.poses[0].position.y:.2f}, Z={msg.poses[0].position.z:.2f}")
                print(f"   最后一个航点: X={msg.poses[-1].position.x:.2f}, Y={msg.poses[-1].position.y:.2f}, Z={msg.poses[-1].position.z:.2f}")
                self.last_path_count = path_count
                
    def send_mission_start_signal(self):
        """发送任务启动信号"""
        try:
            import subprocess
            cmd = "rostopic pub /mission_start_signal std_msgs/Bool 'data: true' --once"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                print("✅ 任务启动信号发送成功")
            else:
                print(f"❌ 任务启动信号发送失败: {result.stderr}")
                
        except Exception as e:
            print(f"❌ 发送任务启动信号时出错: {e}")
            
    def run(self):
        """运行自动发布器"""
        rate = rospy.Rate(1)  # 1Hz
        
        while not rospy.is_shutdown():
            try:
                rate.sleep()
            except KeyboardInterrupt:
                print("\n\n自动路径发布器停止")
                break
            except Exception as e:
                print(f"自动发布器出错: {e}")
                break

def main():
    try:
        publisher = AutoPathPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        print("自动路径发布器已停止")
    except Exception as e:
        print(f"启动自动路径发布器时出错: {e}")

if __name__ == '__main__':
    main() 
