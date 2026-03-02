# NoFlyZone消息定义
# 用于表示禁飞区域信息

# 标准ROS消息头
Header header

# 禁飞区域类型常量
uint8 NO_FLY_ZONE_RECTANGLE=1
uint8 NO_FLY_ZONE_CIRCLE=2
uint8 NO_FLY_ZONE_POLYGON=3
uint8 NO_FLY_ZONE_GRID_POINTS=4

# 消息字段
uint8 zone_type          # 禁飞区域类型
uint32 zone_id           # 禁飞区域ID
uint8 priority           # 优先级（1-10，10为最高）
bool is_active           # 是否激活
geometry_msgs/Point[] points  # 禁飞区域边界点
float64[] parameters     # 额外参数（如半径、高度等）
string description       # 禁飞区域描述

# 禁飞区域属性
float64 min_height       # 最小高度限制
float64 max_height       # 最大高度限制
bool is_temporary        # 是否为临时禁飞区
time valid_until         # 有效期截止时间 