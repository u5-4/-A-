# BoundingBox消息定义
# 用于表示单个检测框的信息

# 检测框的边界坐标
int64 xmin
int64 ymin
int64 xmax
int64 ymax

# 检测框中心点坐标
float64 center_x
float64 center_y

# 检测框的宽度和高度
float64 width
float64 height

# 检测对象的类别名称
string Class

# 检测对象的ID（用于跟踪）
int64 id

# 检测置信度（0.0-1.0）
float64 probability

# 检测标志（用于控制是否处理该检测结果）
bool flag

# 检测框的面积
float64 area

# 检测时间戳
time timestamp 