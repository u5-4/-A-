# BoundingBoxes消息定义
# 用于表示多个检测框的集合

# 标准ROS消息头
Header header

# 图像头信息（用于关联检测结果与图像）
Header image_header

# 检测框数组
BoundingBox[] bounding_boxes

# 检测统计信息
int32 total_detections
float64 average_confidence
string detection_model

# 检测时间戳
time detection_timestamp 