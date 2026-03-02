#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO识别诊断测试脚本
用于诊断YOLO模型是否存在识别过少或识别过多的问题
"""

import rospy
import json
import time
from collections import defaultdict, Counter
from yolov8_ros_msgs.msg import BoundingBoxes, BoundingBox
from std_msgs.msg import String
import numpy as np
from datetime import datetime

class YOLODetectionDiagnosis:
    def __init__(self):
        rospy.init_node('yolo_detection_diagnosis', anonymous=True)
        
        # 订阅YOLO检测结果
        self.bboxes_sub = rospy.Subscriber('/bboxes_pub', BoundingBoxes, self.bboxes_callback)
        
        # 发布诊断结果
        self.diagnosis_pub = rospy.Publisher('/yolo_diagnosis_report', String, queue_size=10)
        
        # 诊断数据存储
        self.detection_history = []
        self.frame_count = 0
        self.start_time = time.time()
        
        # 统计变量
        self.total_detections = 0
        self.animal_type_counts = defaultdict(int)
        self.confidence_distribution = []
        self.duplicate_detections = 0
        self.low_confidence_detections = 0
        self.high_confidence_detections = 0
        
        # 配置参数
        self.confidence_threshold = 0.5
        self.duplicate_threshold = 0.8
        self.analysis_duration = 30
        self.report_interval = 5
        
        # 定时器
        self.report_timer = rospy.Timer(rospy.Duration(self.report_interval), self.publish_diagnosis_report)
        
        print("�� YOLO识别诊断系统启动")
        print(f"�� 分析持续时间: {self.analysis_duration}秒")
        print(f"�� 报告间隔: {self.report_interval}秒")
        print(f"�� 置信度阈值: {self.confidence_threshold}")
        print("=" * 60)

    def bboxes_callback(self, msg):
        """处理YOLO检测结果"""
        self.frame_count += 1
        current_time = time.time()
        
        # 记录检测信息
        frame_data = {
            'timestamp': current_time,
            'frame_id': self.frame_count,
            'detections': [],
            'total_objects': len(msg.bounding_boxes)
        }
        
        # 分析每个检测框
        for bbox in msg.bounding_boxes:
            detection_info = {
                'class': bbox.Class,
                'confidence': bbox.probability,
                'x': bbox.xmin,
                'y': bbox.ymin,
                'width': bbox.xmax - bbox.xmin,
                'height': bbox.ymax - bbox.ymin,
                'area': (bbox.xmax - bbox.xmin) * (bbox.ymax - bbox.ymin)
            }
            
            frame_data['detections'].append(detection_info)
            
            # 更新统计信息
            self.total_detections += 1
            self.animal_type_counts[bbox.Class] += 1
            self.confidence_distribution.append(bbox.probability)
            
            # 置信度分类
            if bbox.probability < self.confidence_threshold:
                self.low_confidence_detections += 1
            else:
                self.high_confidence_detections += 1
        
        self.detection_history.append(frame_data)
        
        # 检测重复识别
        self.detect_duplicates(frame_data['detections'])
        
        # 实时显示检测信息
        if self.frame_count % 10 == 0:
            self.print_realtime_stats()

    def detect_duplicates(self, detections):
        """检测重复识别"""
        if len(detections) < 2:
            return
        
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                iou = self.calculate_iou(detections[i], detections[j])
                if iou > self.duplicate_threshold:
                    self.duplicate_detections += 1
                    print(f"⚠️  检测到重复识别: {detections[i]['class']} 和 {detections[j]['class']}, IOU: {iou:.3f}")

    def calculate_iou(self, bbox1, bbox2):
        """计算两个边界框的IOU"""
        x1 = max(bbox1['x'], bbox2['x'])
        y1 = max(bbox1['y'], bbox2['y'])
        x2 = min(bbox1['x'] + bbox1['width'], bbox2['x'] + bbox2['width'])
        y2 = min(bbox1['y'] + bbox1['height'], bbox2['y'] + bbox2['height'])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = bbox1['area'] + bbox2['area'] - intersection
        
        return intersection / union if union > 0 else 0.0

    def print_realtime_stats(self):
        """打印实时统计信息"""
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n�� 实时统计 (帧 {self.frame_count})")
        print(f"⏱️  运行时间: {elapsed_time:.1f}s")
        print(f"�� FPS: {fps:.1f}")
        print(f"�� 总检测数: {self.total_detections}")
        print(f"�� 平均每帧检测数: {self.total_detections/self.frame_count:.2f}")
        print(f"�� 动物类型分布: {dict(self.animal_type_counts)}")
        print(f"⚠️  重复检测: {self.duplicate_detections}")
        print(f"�� 低置信度检测: {self.low_confidence_detections}")
        print(f"�� 高置信度检测: {self.high_confidence_detections}")

    def analyze_detection_patterns(self):
        """分析检测模式"""
        if not self.detection_history:
            return {}
        
        analysis = {}
        
        # 检测数量趋势
        detection_counts = [frame['total_objects'] for frame in self.detection_history]
        analysis['avg_detections_per_frame'] = np.mean(detection_counts)
        analysis['max_detections_per_frame'] = np.max(detection_counts)
        analysis['min_detections_per_frame'] = np.min(detection_counts)
        analysis['detection_variance'] = np.var(detection_counts)
        
        # 置信度分析
        if self.confidence_distribution:
            analysis['avg_confidence'] = np.mean(self.confidence_distribution)
            analysis['confidence_std'] = np.std(self.confidence_distribution)
            analysis['confidence_range'] = (np.min(self.confidence_distribution), np.max(self.confidence_distribution))
        
        # 动物类型分析
        analysis['animal_types'] = dict(self.animal_type_counts)
        analysis['most_common_animal'] = max(self.animal_type_counts.items(), key=lambda x: x[1]) if self.animal_type_counts else None
        
        # 检测稳定性分析
        zero_detection_frames = sum(1 for frame in self.detection_history if frame['total_objects'] == 0)
        analysis['zero_detection_rate'] = zero_detection_frames / len(self.detection_history)
        
        return analysis

    def diagnose_issues(self, analysis):
        """诊断识别问题"""
        issues = []
        recommendations = []
        
        # 检测过少的问题
        if analysis['avg_detections_per_frame'] < 0.5:
            issues.append("❌ 检测过少: 平均每帧检测数过低")
            recommendations.append("建议: 降低置信度阈值或检查模型训练数据")
        
        if analysis['zero_detection_rate'] > 0.3:
            issues.append("❌ 空检测率过高: 大量帧没有检测到任何目标")
            recommendations.append("建议: 检查摄像头设置、光照条件或模型性能")
        
        # 检测过多的问题
        if analysis['avg_detections_per_frame'] > 5:
            issues.append("⚠️  检测过多: 平均每帧检测数过高")
            recommendations.append("建议: 提高置信度阈值或检查是否存在误检")
        
        if self.duplicate_detections > self.total_detections * 0.1:
            issues.append("⚠️  重复检测过多: 可能存在同一目标的多次检测")
            recommendations.append("建议: 调整NMS参数或检查模型输出")
        
        # 置信度问题
        if analysis['avg_confidence'] < 0.6:
            issues.append("⚠️  平均置信度较低: 模型对检测结果不够确信")
            recommendations.append("建议: 检查模型训练质量或调整后处理参数")
        
        if self.low_confidence_detections > self.high_confidence_detections:
            issues.append("⚠️  低置信度检测过多: 可能存在大量误检")
            recommendations.append("建议: 提高置信度阈值")
        
        # 检测稳定性问题
        if analysis['detection_variance'] > 2.0:
            issues.append("⚠️  检测数量波动较大: 检测不够稳定")
            recommendations.append("建议: 检查环境变化或模型鲁棒性")
        
        return issues, recommendations

    def publish_diagnosis_report(self, event):
        """发布诊断报告"""
        analysis = self.analyze_detection_patterns()
        issues, recommendations = self.diagnose_issues(analysis)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'frame_count': self.frame_count,
            'total_detections': self.total_detections,
            'analysis': analysis,
            'issues': issues,
            'recommendations': recommendations,
            'animal_type_counts': dict(self.animal_type_counts),
            'confidence_stats': {
                'avg': np.mean(self.confidence_distribution) if self.confidence_distribution else 0,
                'std': np.std(self.confidence_distribution) if self.confidence_distribution else 0,
                'low_confidence_count': self.low_confidence_detections,
                'high_confidence_count': self.high_confidence_detections
            }
        }
        
        # 发布报告
        report_msg = String()
        report_msg.data = json.dumps(report, ensure_ascii=False)
        self.diagnosis_pub.publish(report_msg)
        
        # 打印报告
        self.print_diagnosis_report(report)

    def print_diagnosis_report(self, report):
        """打印诊断报告"""
        print("\n" + "=" * 60)
        print("�� YOLO识别诊断报告")
        print("=" * 60)
        
        print(f"�� 基本统计:")
        print(f"   总帧数: {report['frame_count']}")
        print(f"   总检测数: {report['total_detections']}")
        print(f"   平均每帧检测数: {report['analysis']['avg_detections_per_frame']:.2f}")
        print(f"   空检测率: {report['analysis']['zero_detection_rate']:.2%}")
        
        print(f"\n�� 动物类型分布:")
        for animal_type, count in report['animal_type_counts'].items():
            print(f"   {animal_type}: {count}")
        
        print(f"\n�� 置信度统计:")
        print(f"   平均置信度: {report['confidence_stats']['avg']:.3f}")
        print(f"   置信度标准差: {report['confidence_stats']['std']:.3f}")
        print(f"   低置信度检测: {report['confidence_stats']['low_confidence_count']}")
        print(f"   高置信度检测: {report['confidence_stats']['high_confidence_count']}")
        
        print(f"\n⚠️  检测到的问题:")
        if report['issues']:
            for issue in report['issues']:
                print(f"   {issue}")
        else:
            print("   ✅ 未发现明显问题")
        
        print(f"\n�� 建议:")
        if report['recommendations']:
            for rec in report['recommendations']:
                print(f"   {rec}")
        else:
            print("   ✅ 当前设置合理")
        
        print("=" * 60)

    def save_detection_data(self, filename=None):
        """保存检测数据到文件"""
        if filename is None:
            filename = f"yolo_detection_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            'detection_history': self.detection_history,
            'statistics': {
                'total_detections': self.total_detections,
                'animal_type_counts': dict(self.animal_type_counts),
                'confidence_distribution': self.confidence_distribution,
                'duplicate_detections': self.duplicate_detections,
                'frame_count': self.frame_count
            },
            'analysis': self.analyze_detection_patterns()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"�� 检测数据已保存到: {filename}")

def main():
    try:
        diagnosis = YOLODetectionDiagnosis()
        
        print("�� 开始YOLO识别诊断...")
        print("按 Ctrl+C 停止诊断")
        
        # 运行诊断
        rospy.spin()
        
    except KeyboardInterrupt:
        print("\n⏹️  诊断停止")
        # 保存数据
        diagnosis.save_detection_data()
        
    except rospy.ROSInterruptException:
        print("诊断被中断")
    except Exception as e:
        print(f"诊断过程中发生错误: {e}")

if __name__ == "__main__":
    main() 
