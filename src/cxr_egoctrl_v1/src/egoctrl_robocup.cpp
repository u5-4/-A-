// 数学常量定义
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// ========== 头文件区 ==========
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/Joy.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/CommandLong.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/PositionTarget.h>
#include <mavros_msgs/RCIn.h>
#include <sensor_msgs/BatteryState.h>
#include "quadrotor_msgs/PositionCommand.h"
#include "quadrotor_msgs/TakeoffLand.h"
#include "quadrotor_msgs/GoalSet.h"
#include <nav_msgs/Odometry.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <Eigen/Core>
#include <mutex>
#include <algorithm>  // 添加algorithm头文件，支持std::min和std::max
#include <yolov8_ros_msgs/BoundingBoxes.h>
#include <geometry_msgs/PoseArray.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
// #include <opencv2/opencv.hpp>
// #include <cv_bridge/cv_bridge.h>

// ========== 全局参数/常量区 ==========
const float TAKEOFF_HEIGHT = 1.2;      // 起飞高度
const float THRESHOLD = 0.3;           // 到点判定距离阈值：减小到0.3米，提高精度
const float KP = 0.15;                 // P控制器增益：降低到0.15，提高稳定性
const float MAX_SPEED = 0.3;           // 最大速度：降低到0.3m/s，确保稳定飞行
const float HOVER_TIME = 1.0;          // 航点悬停时间：1秒，消除P控制误差
const float PI = 3.14159265;

// 高级PID控制参数 - 优化版本（针对位置偏差问题）
const float PID_KP_X = 1.2;        // 从0.8增加到1.2，进一步提高X轴响应速度
const float PID_KP_Y = 1.2;        // 从0.8增加到1.2，进一步提高Y轴响应速度  
const float PID_KP_Z = 1.0;        // 从0.6增加到1.0，进一步提高Z轴响应速度
const float PID_KI_X = 0.0;            // X方向I增益（暂不使用）
const float PID_KI_Y = 0.0;            // Y方向I增益（暂不使用）
const float PID_KI_Z = 0.0;            // Z方向I增益（暂不使用）
const float PID_KD_X = 0.0;            // X方向D增益（暂不使用）
const float PID_KD_Y = 0.0;            // Y方向D增益（暂不使用）
const float PID_KD_Z = 0.0;            // Z方向D增益（暂不使用）
const float SPEED_SCALE = 0.8;     // 从0.4增加到0.8，提高整体速度

//相机内参（标定后）
const double FX = 459.758142;  // 焦距x (像素)
const double FY = 459.609932;  // 焦距y (像素)
const double CX = 326.465004;  // 光心x (像素)
const double CY = 233.811816;  // 光心y (像素)

//相机外参（相对于无人机机体坐标系）
// 坐标系说明：
// - 机头方向：X轴正方向（向前）
// - 机头左边：Y轴正方向（向左）
// - 向上：Z轴正方向
const double CAMERA_X = 0.08;  // 相机在无人机前方9.9cm（9.8-10.0cm范围的中点）
const double CAMERA_Y = 0.0;    // 相机在无人机中心线上
const double CAMERA_Z = 0.05; // 相机在无人机中心上方2.25cm（2.0-2.5cm范围的中点）
const double CAMERA_ROLL = 0.0;     // 无横滚
const double CAMERA_PITCH = -90.0;    //  俯仰角：-90°（向下垂直）
const double CAMERA_YAW = 0.0;      // 无偏航

// ========== 对齐坐标发布者 ==========
ros::Publisher aligned_center_pub;  // 发布对齐后的中心坐标
bool alignment_completed = false;    // 对齐完成标志

// ========== 飞行轨迹记录与对比 ==========
struct FlightPoint {
    double x, y, z;
    ros::Time timestamp;
    int waypoint_index;
    std::string grid_position;
};

std::vector<FlightPoint> actual_flight_trajectory;  // 实际飞行轨迹
std::vector<Eigen::Vector3d> planned_path_trajectory;  // 规划路径轨迹
bool trajectory_recording_enabled = false;  // 轨迹记录开关
ros::Publisher trajectory_pub;  // 发布轨迹对比信息
ros::Subscriber planning_stats_sub;  // 订阅规划统计信息

// ========== 可调PID参数（运行时可通过话题调整） ==========
float current_pid_kp_x = PID_KP_X;
float current_pid_kp_y = PID_KP_Y;
float current_pid_kp_z = PID_KP_Z;
float current_speed_scale = SPEED_SCALE;
ros::Subscriber pid_params_sub;  // 订阅PID参数调整话题

// ========== 目标点数组（使用Eigen::Vector3d获得更好的数学运算能力） ==========
// 网格路径点（新坐标系：X轴平行于B列（向前），Y轴平行于A列（向左），单位：米）
// 注意：A9B1为原点(0,0)，X轴：B1到B7（向前），Y轴：A9到A1（向左）
std::vector<Eigen::Vector3d> positions = { 
    // 从起飞点A9B1开始，按网格顺序飞行
    Eigen::Vector3d(0.0, 0.0, TAKEOFF_HEIGHT),      // A9B1: 起飞点（原点）
    Eigen::Vector3d(0.0, 0.5, TAKEOFF_HEIGHT),      // A8B1
    Eigen::Vector3d(0.0, 1.0, TAKEOFF_HEIGHT),      // A7B1
    Eigen::Vector3d(0.0, 1.5, TAKEOFF_HEIGHT),      // A6B1
    Eigen::Vector3d(0.0, 2.0, TAKEOFF_HEIGHT),      // A5B1
    Eigen::Vector3d(0.0, 2.5, TAKEOFF_HEIGHT),      // A4B1
    Eigen::Vector3d(0.0, 3.0, TAKEOFF_HEIGHT),      // A3B1
    Eigen::Vector3d(0.0, 3.5, TAKEOFF_HEIGHT),      // A2B1
    Eigen::Vector3d(0.0, 4.0, TAKEOFF_HEIGHT),      // A1B1
    
    Eigen::Vector3d(0.5, 4.0, TAKEOFF_HEIGHT),      // A1B2
    Eigen::Vector3d(0.5, 3.5, TAKEOFF_HEIGHT),      // A2B2
    Eigen::Vector3d(0.5, 3.0, TAKEOFF_HEIGHT),      // A3B2
    Eigen::Vector3d(0.5, 2.5, TAKEOFF_HEIGHT),      // A4B2
    Eigen::Vector3d(0.5, 2.0, TAKEOFF_HEIGHT),      // A5B2
    Eigen::Vector3d(0.5, 1.5, TAKEOFF_HEIGHT),      // A6B2
    Eigen::Vector3d(0.5, 1.0, TAKEOFF_HEIGHT),      // A7B2
    Eigen::Vector3d(0.5, 0.5, TAKEOFF_HEIGHT),      // A8B2
    Eigen::Vector3d(0.5, 0.0, TAKEOFF_HEIGHT),      // A9B2
    
    Eigen::Vector3d(1.0, 0.0, TAKEOFF_HEIGHT),      // A9B3
    Eigen::Vector3d(1.0, 0.5, TAKEOFF_HEIGHT),      // A8B3
    Eigen::Vector3d(1.0, 1.0, TAKEOFF_HEIGHT),      // A7B3
    Eigen::Vector3d(1.0, 1.5, TAKEOFF_HEIGHT),      // A6B3
    Eigen::Vector3d(1.0, 2.0, TAKEOFF_HEIGHT),      // A5B3
    Eigen::Vector3d(1.0, 2.5, TAKEOFF_HEIGHT),      // A4B3
    Eigen::Vector3d(1.0, 3.0, TAKEOFF_HEIGHT),      // A3B3
    Eigen::Vector3d(1.0, 3.5, TAKEOFF_HEIGHT),      // A2B3
    Eigen::Vector3d(1.0, 4.0, TAKEOFF_HEIGHT),      // A1B3
    
    Eigen::Vector3d(1.5, 4.0, TAKEOFF_HEIGHT),      // A1B4
    Eigen::Vector3d(1.5, 3.5, TAKEOFF_HEIGHT),      // A2B4
    Eigen::Vector3d(1.5, 3.0, TAKEOFF_HEIGHT),      // A3B4
    Eigen::Vector3d(1.5, 2.5, TAKEOFF_HEIGHT),      // A4B4
    Eigen::Vector3d(1.5, 2.0, TAKEOFF_HEIGHT),      // A5B4
    Eigen::Vector3d(1.5, 1.5, TAKEOFF_HEIGHT),      // A6B4
    Eigen::Vector3d(1.5, 1.0, TAKEOFF_HEIGHT),      // A7B4
    Eigen::Vector3d(1.5, 0.5, TAKEOFF_HEIGHT),      // A8B4
    Eigen::Vector3d(1.5, 0.0, TAKEOFF_HEIGHT),      // A9B4
    
    Eigen::Vector3d(2.0, 0.0, TAKEOFF_HEIGHT),      // A9B5
    Eigen::Vector3d(2.0, 0.5, TAKEOFF_HEIGHT),      // A8B5
    Eigen::Vector3d(2.0, 1.0, TAKEOFF_HEIGHT),      // A7B5
    Eigen::Vector3d(2.0, 1.5, TAKEOFF_HEIGHT),      // A6B5
    Eigen::Vector3d(2.0, 2.0, TAKEOFF_HEIGHT),      // A5B5
    Eigen::Vector3d(2.0, 2.5, TAKEOFF_HEIGHT),      // A4B5
    Eigen::Vector3d(2.0, 3.0, TAKEOFF_HEIGHT),      // A3B5
    Eigen::Vector3d(2.0, 3.5, TAKEOFF_HEIGHT),      // A2B5
    Eigen::Vector3d(2.0, 4.0, TAKEOFF_HEIGHT),      // A1B5
    
    Eigen::Vector3d(2.5, 4.0, TAKEOFF_HEIGHT),      // A1B6
    Eigen::Vector3d(2.5, 3.5, TAKEOFF_HEIGHT),      // A2B6
    Eigen::Vector3d(2.5, 3.0, TAKEOFF_HEIGHT),      // A3B6
    Eigen::Vector3d(2.5, 2.5, TAKEOFF_HEIGHT),      // A4B6
    Eigen::Vector3d(2.5, 2.0, TAKEOFF_HEIGHT),      // A5B6
    Eigen::Vector3d(2.5, 1.5, TAKEOFF_HEIGHT),      // A6B6
    Eigen::Vector3d(2.5, 1.0, TAKEOFF_HEIGHT),      // A7B6
    Eigen::Vector3d(2.5, 0.5, TAKEOFF_HEIGHT),      // A8B6
    Eigen::Vector3d(2.5, 0.0, TAKEOFF_HEIGHT),      // A9B6
    
    Eigen::Vector3d(3.0, 0.0, TAKEOFF_HEIGHT),      // A9B7
    Eigen::Vector3d(3.0, 0.5, TAKEOFF_HEIGHT),      // A8B7
    Eigen::Vector3d(3.0, 1.0, TAKEOFF_HEIGHT),      // A7B7
    Eigen::Vector3d(3.0, 1.5, TAKEOFF_HEIGHT),      // A6B7
    Eigen::Vector3d(3.0, 2.0, TAKEOFF_HEIGHT),      // A5B7
    Eigen::Vector3d(3.0, 2.5, TAKEOFF_HEIGHT),      // A4B7
    Eigen::Vector3d(3.0, 3.0, TAKEOFF_HEIGHT),      // A3B7
    Eigen::Vector3d(3.0, 3.5, TAKEOFF_HEIGHT),      // A2B7
    Eigen::Vector3d(3.0, 4.0, TAKEOFF_HEIGHT)       // A1B7: 终点
};

// 动态路径点（从路径规划器接收）
std::vector<Eigen::Vector3d> dynamic_positions;
bool use_dynamic_path = false;  // 是否使用动态路径

// ========== 航点管理全局变量 ==========
int global_current_waypoint = 0;
bool global_waypoint_entered = false;
ros::Time global_waypoint_start_time;  // 将在main()中初始化
bool global_waypoint_started = false;
bool global_first_enter = true;

// ========== 降落位置控制全局变量 ==========
Eigen::Vector3d landing_target_position(0.0, 0.0, 0.0);  // 降落目标位置（原点）
bool landing_position_set = false;  // 是否已设置降落位置
float landing_position_tolerance = 0.5;  // 降落位置容差（米）
float landing_horizontal_speed = 0.3;  // 降落时水平移动速度

// ========== 返航起飞点悬停控制全局变量 ==========
bool return_takeoff_point_hover_enabled = true;  // 是否启用返航起飞点悬停
float return_takeoff_point_tolerance = 0.3;  // 返航起飞点位置容差（米）
float return_takeoff_point_hover_time = 2.0;  // 返航起飞点悬停时间（秒）
ros::Time return_takeoff_point_hover_start_time;  // 返航起飞点悬停开始时间
bool return_takeoff_point_hover_started = false;  // 是否已开始返航起飞点悬停
bool return_takeoff_point_hover_completed = false;  // 返航起飞点悬停是否完成
Eigen::Vector3d return_takeoff_point(0.0, 0.0, TAKEOFF_HEIGHT);  // 返航起飞点位置A9B1(0,0) - 高度为TAKEOFF_HEIGHT

// ========== MAVROS/飞控相关全局变量 ==========
#define VELOCITY2D_CONTROL 0b101111000111
#define POSITION2D_CONTROL 0b101111111000
#define HYBRID_VEL_CONTROL 0b101111000111
#define HYBRID_POS_CONTROL 0b101111111011
unsigned short velocity_mask = VELOCITY2D_CONTROL;
unsigned short pisition_mask = POSITION2D_CONTROL;
mavros_msgs::PositionTarget current_goal;
mavros_msgs::RCIn rc;
sensor_msgs::BatteryState bat;
ros::ServiceClient arming_client;
ros::ServiceClient command_client;
ros::ServiceClient set_mode_client;
mavros_msgs::SetMode offb_set_mode;
mavros_msgs::CommandBool arm_cmd;
ros::Publisher local_pos_pub;
ros::Publisher pos_cmd_pub;
ros::Publisher mission_start_pub;  // 任务启动信号发布者
ros::Publisher animal_location_pub;  // 动物位置信息发布者
ros::Publisher count_allow_pub;      // 计数允许信号发布者
ros::Time last_request;
ros::Time mission_hover_time;
ros::Time mission_cali_time;
ros::Time mission_land_time;
double volt{0.0};
double percentage{0.0};
ros::Time rcv_stamp;
ros::Time last_print_t;
int rc_value, flag = 0, flag1 = 0;
nav_msgs::Odometry position_msg;
Eigen::Vector3d odom_v;
Eigen::Vector3d odom_p;
Eigen::Vector3d point1;
geometry_msgs::PoseStamped target_pos;
mavros_msgs::State current_state;
float position_x, position_y, position_z,  current_yaw, targetpos_x, targetpos_y;
float ego_pos_x, ego_pos_y, ego_pos_z, ego_vel_x, ego_vel_y, ego_vel_z, ego_a_x, ego_a_y, ego_a_z, ego_yaw, ego_yaw_rate;
bool receive = false, land = false, takeoff = false;
bool startmission = false;
bool point1_send = false;
bool point1_finished = false;
ros::Time target_time_;
ros::Time toufang_time;

quadrotor_msgs::PositionCommand ego;  // 添加 ego 声明
quadrotor_msgs::TakeoffLand takeoffland;  // 添加 takeoffland 声明
void publish_aligned_center(float world_x, float world_y, float depth, int qr_id);
void pixel_to_world_coordinates_advanced(float pixel_x, float pixel_y, float depth, 
                                        float& world_x, float& world_y, float current_yaw);
void planned_path_callback(const geometry_msgs::PoseArray::ConstPtr& msg);
void return_path_callback(const geometry_msgs::PoseArray::ConstPtr& msg);  // 返航路径回调
void planning_stats_callback(const std_msgs::String::ConstPtr& msg);
void record_flight_trajectory(int waypoint_index, const std::string& grid_position);
void analyze_trajectory_accuracy();
void pid_params_callback(const std_msgs::String::ConstPtr& msg);  // PID参数调整回调
void animal_stats_callback(const std_msgs::String::ConstPtr& msg);  // 动物统计信息回调
void parse_animal_counts(const std::string& counts_str, std::map<std::string, int>& counts_map);  // 解析动物计数
void make_flight_decisions();  // 基于统计信息进行飞行决策

// ========== 坐标转换和重复航点检测函数 ==========
std::string world_to_grid_position(float x, float y);  // 世界坐标转网格坐标
void analyze_duplicate_waypoints(const geometry_msgs::PoseArray::ConstPtr& path_msg);  // 分析重复航点
void publish_animal_location(const std::string& grid_pos, const std::string& animal_type);  // 发布动物位置信息


// ========== 动物检测相关全局变量 ==========
yolov8_ros_msgs::BoundingBoxes bboxes;  // 动物检测结果
bool animal_detected = false;                        // 动物检测标志
std::string detected_animal_type = "";               // 检测到的动物类型
float animal_confidence = 0.0;                       // 动物检测置信度

// ========== 动物统计相关全局变量 ==========
struct AnimalStatistics {
    int total_count = 0;
    std::map<std::string, int> animal_counts;
    std::map<std::string, int> current_frame_counts;
    std::string detection_time;
    bool data_valid = false;
};
AnimalStatistics animal_stats;  // 动物统计信息

// ========== 重复航点检测相关变量 ==========
std::map<std::string, int> waypoint_visit_count;  // 记录每个航点的访问次数
std::set<std::string> duplicate_waypoints;        // 重复航点集合
std::string current_grid_position = "";           // 当前网格位置
std::string last_grid_position = "";              // 上一个网格位置
bool is_at_duplicate_waypoint = false;            // 是否在重复航点
bool animal_detected_at_duplicate = false;        // 是否在重复航点检测到动物

// ========== 转头控制器相关全局变量 ==========
ros::Time yaw_start_time;
bool yaw_controller_active = false;
double yaw_initial_angle = 0.0;
double yaw_target_angle = 0.0;
double yaw_rate = 0.5;  // 转头速度（弧度/秒）

// ========== TF2坐标转换相关变量 ==========
tf2_ros::Buffer tf_buffer;
tf2_ros::TransformListener* tf_listener = nullptr;
bool tf_available = false;

// ========== 返航路径相关变量 ==========
std::vector<Eigen::Vector3d> return_path;  // 返航路径
// 返航路径相关变量（保持向后兼容，但不再使用）
bool return_path_received = false;         // 是否收到返航路径（兼容性）
bool return_mission_started = false;       // 返航任务是否已开始（兼容性）
int return_waypoint_index = 0;             // 返航航点索引（兼容性）


// ========== 状态机相关变量 ==========
enum State_t {
    INIT, MISSION, HOVER, TAKEOFF, LANDING, AUTOLAND
};
State_t state;
enum Mission_State_t {
    MISSION_INIT, MISSION_TAKEOFF, MISSION_LAND, MISSION_HOVER,
    MISSION_POINT1, MISSION_HOVER_POINT, MISSION_POINT2, MISSION_HOVER_POINT2,
    MISSION_QR_ALIGNMENT,  // 二维码对齐状态
    MISSION_SEARCH_QR,     // 搜索二维码状态
    MISSION_HOVER_BEFORE_LAND  // 悬停后降落状态
};
Mission_State_t mission_state;
bool mission_failed = false;

// ========== 高度零点补偿全局变量 ==========
float height_offset = 0.0;
bool height_offset_set = false;

// 高度零点重置函数
void reset_height_offset() {
    height_offset = position_z;
    height_offset_set = true;
    ROS_WARN("手动重置高度零点！当前绝对高度: %.3f, 相对高度设为0", position_z);
}

// ========== 降落位置控制函数 ==========
void set_landing_target_position() {
    if (!landing_position_set) {
        // 始终使用A9B1(0,0)作为降落目标位置
        landing_target_position = Eigen::Vector3d(0.0, 0.0, 0.0);
        landing_position_set = true;
        ROS_INFO("设置降落目标位置: A9B1(0,0) - (%.2f, %.2f, %.2f)", 
                 landing_target_position.x(), landing_target_position.y(), landing_target_position.z());
    }
}

bool should_return_to_landing_position() {
    if (!landing_position_set) {
        return false;
    }
    
    // 计算当前位置到降落目标的水平距离
    float dx = position_x - landing_target_position.x();
    float dy = position_y - landing_target_position.y();
    float horizontal_distance = sqrt(dx*dx + dy*dy);
    
    return horizontal_distance > landing_position_tolerance;
}

void calculate_landing_velocity(float& vx, float& vy, float& vz) {
    if (!landing_position_set) {
        vx = 0.0;
        vy = 0.0;
        vz = -0.2;  // 默认下降速度
        return;
    }
    
    // 计算到降落目标的位置差
    float dx = landing_target_position.x() - position_x;
    float dy = landing_target_position.y() - position_y;
    float horizontal_distance = sqrt(dx*dx + dy*dy);
    
    if (horizontal_distance > landing_position_tolerance) {
        // 需要水平移动，计算方向向量
        float direction_x = dx / horizontal_distance;
        float direction_y = dy / horizontal_distance;
        
        // 限制水平速度
        vx = direction_x * landing_horizontal_speed;
        vy = direction_y * landing_horizontal_speed;
        vz = -0.1;  // 缓慢下降
        
        ROS_INFO_THROTTLE(2.0, "降落位置控制: 距离=%.2f, 方向=(%.2f,%.2f), 速度=(%.2f,%.2f,%.2f)", 
                         horizontal_distance, direction_x, direction_y, vx, vy, vz);
    } else {
        // 已在降落位置，只进行垂直下降
        vx = 0.0;
        vy = 0.0;
        vz = -0.2;
        
        ROS_INFO_THROTTLE(2.0, "已在降落位置，垂直下降: 距离=%.2f", horizontal_distance);
    }
}

// ========== 返航起飞点悬停控制函数 ==========
bool is_at_return_takeoff_point() {
    // 检查是否到达返航起飞点A9B1(0,0) - 高度为TAKEOFF_HEIGHT
    float dx = position_x - return_takeoff_point.x();
    float dy = position_y - return_takeoff_point.y();
    float dz = position_z - return_takeoff_point.z();
    
    float horizontal_distance = sqrt(dx*dx + dy*dy);
    float vertical_distance = abs(dz);
    
    // 水平距离和垂直距离都在容差范围内
    bool at_position = (horizontal_distance <= return_takeoff_point_tolerance && 
                       vertical_distance <= return_takeoff_point_tolerance);
    
    if (at_position) {
        ROS_INFO_THROTTLE(2.0, "�� 到达返航起飞点A9B1(0,0): 水平距离=%.2fm, 垂直距离=%.2fm", 
                         horizontal_distance, vertical_distance);
    }
    
    return at_position;
}

void start_return_takeoff_point_hover() {
    if (!return_takeoff_point_hover_started) {
        return_takeoff_point_hover_start_time = ros::Time::now();
        return_takeoff_point_hover_started = true;
        return_takeoff_point_hover_completed = false;
        ROS_INFO("�� 开始返航起飞点悬停: A9B1(0,0) 高度%.1fm，悬停时间%.1f秒", 
                 TAKEOFF_HEIGHT, return_takeoff_point_hover_time);
    }
}

bool is_return_takeoff_point_hover_completed() {
    if (!return_takeoff_point_hover_started) {
        return false;
    }
    
    ros::Time current_time = ros::Time::now();
    float hover_duration = (current_time - return_takeoff_point_hover_start_time).toSec();
    
    if (hover_duration >= return_takeoff_point_hover_time) {
        if (!return_takeoff_point_hover_completed) {
            return_takeoff_point_hover_completed = true;
            ROS_INFO("✅ 返航起飞点悬停完成: 悬停时间%.1f秒，开始降落", hover_duration);
        }
        return true;
    }
    
    // 显示悬停进度
    ROS_INFO_THROTTLE(1.0, "返航起飞点悬停中: %.1f/%.1f秒", 
                     hover_duration, return_takeoff_point_hover_time);
    return false;
}

void reset_return_takeoff_point_hover() {
    return_takeoff_point_hover_started = false;
    return_takeoff_point_hover_completed = false;
    ROS_INFO("重置返航起飞点悬停状态");
}

// 限幅函数
float limit(float value, float min_value, float max_value) {
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
}

// ========== 转头控制函数 ==========
// 开始转头
void start_yaw_control(double target_yaw, double rate, double current_yaw) {
    yaw_start_time = ros::Time::now();
    yaw_controller_active = true;
    yaw_initial_angle = current_yaw;
    yaw_target_angle = target_yaw;
    yaw_rate = rate;
    
    // 计算角度差（处理角度环绕）
    double yaw_diff = target_yaw - current_yaw;
    while (yaw_diff > M_PI) yaw_diff -= 2 * M_PI;
    while (yaw_diff < -M_PI) yaw_diff += 2 * M_PI;
    
    double total_time = fabs(yaw_diff) / rate;
    
    ROS_INFO("开始转头控制: 从 %.2f 转到 %.2f, 角度差: %.2f, 速度: %.2f rad/s, 预计时间: %.2f秒", 
             yaw_initial_angle, yaw_target_angle, yaw_diff, yaw_rate, total_time);
}

// 停止转头控制
void stop_yaw_control() {
    if (yaw_controller_active) {
        yaw_controller_active = false;
        ROS_INFO("手动停止转头控制");
    }
}

// 检查是否正在转头
bool is_yaw_controller_active() {
    return yaw_controller_active;
}

// 执行转头控制（返回是否完成）
bool execute_yaw_control(double current_yaw, ros::Publisher& local_pos_pub, 
                        mavros_msgs::PositionTarget& current_goal) {
    if (!yaw_controller_active) {
        return true;    // 如果yaw_controller_active为false，则返回true，停止自旋
    }
    
    // 计算角度差（处理角度环绕）
    double yaw_diff = yaw_target_angle - yaw_initial_angle;
    while (yaw_diff > M_PI) yaw_diff -= 2 * M_PI;
    while (yaw_diff < -M_PI) yaw_diff += 2 * M_PI;
    
    // 计算已经过去的时间
    double elapsed_time = (ros::Time::now() - yaw_start_time).toSec();
    
    // 计算需要的总时间（基于角度差和角速度）
    double total_time = fabs(yaw_diff) / yaw_rate;
    
    ROS_DEBUG_THROTTLE(1.0, "转头控制: 角度差=%.2f, 已用时间=%.2f, 总时间=%.2f, 当前偏航=%.2f", 
                       yaw_diff, elapsed_time, total_time, current_yaw);
    
    // 如果还没完成转头
    if (elapsed_time < total_time) {
        // 计算当前应该的偏航角（线性插值）
        double progress = elapsed_time / total_time;
        double current_target_yaw = yaw_initial_angle + yaw_diff * progress;
        
        // 发布速度指令（保持位置不变，只改变偏航）
        current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
        current_goal.header.stamp = ros::Time::now();
        current_goal.type_mask = 0b101111000111; // 只使用速度和偏航
        
        current_goal.velocity.x = 0.0;
        current_goal.velocity.y = 0.0;
        current_goal.velocity.z = 0.0;
        current_goal.yaw = current_target_yaw;  // 关键：设置目标偏航角
        
        local_pos_pub.publish(current_goal);
        return false; // 还没完成
    } else {
        // 转头完成
        yaw_controller_active = false;
        ROS_INFO("转头控制完成，总用时: %.2f秒", elapsed_time);
        return true;
    }
}

// 粗略飞行控制
bool Rough_flight(const std::vector<Eigen::Vector3d>& positions, int idx, float threshold, float kp, float max_speed)
{
    // 计算当前位置与目标点的距离（用相对高度）
    float dx = positions[idx].x() - position_x;
    float dy = positions[idx].y() - position_y;
    float relative_z = position_z - height_offset;
    float dz = positions[idx].z() - relative_z;
    float distance = sqrt(dx * dx + dy * dy + dz * dz);// 计算当前位置与目标点的距离

    static ros::Time hover_start_time;
    static bool hovering = false;
    ros::Duration hover_timeout(HOVER_TIME); // 悬停判定时间：1秒，消除P控制误差
    static bool last_in_threshold = false;

    // 到达目标点附近
    if (distance < threshold) { //首次进入悬停状态和重新进入悬停状态都会导致重新计时
        if (!hovering || !last_in_threshold) {
            hover_start_time = ros::Time::now();
            hovering = true;
        }
        last_in_threshold = true;

        // 悬停指令：使用相对高度控制，目标高度就是positions[idx].z()
        float dz_hover = positions[idx].z() - relative_z;
        // 优化悬停控制，提高高度稳定性
        float vz_hover = 0.0;
        if (abs(dz_hover) > 0.02) {  // 2cm死区
            vz_hover = limit(dz_hover * 0.3, -0.03, 0.03);  // 适中的控制增益
        }
        
        current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
        current_goal.header.stamp = ros::Time::now();
        current_goal.type_mask = velocity_mask;
        current_goal.velocity.x = 0.0;
        current_goal.velocity.y = 0.0;
        current_goal.velocity.z = vz_hover;
        current_goal.yaw = current_yaw;
        local_pos_pub.publish(current_goal);
        ROS_INFO("航点悬停中: 目标(%f, %f, %f), 当前位置(%.3f, %.3f, %.3f), 相对高度=%.3f, 悬停时间=%.1fs", 
                positions[idx].x(), positions[idx].y(), positions[idx].z(), 
                position_x, position_y, position_z, relative_z, 
                (ros::Time::now() - hover_start_time).toSec());

        // 判断是否稳定悬停足够时间
        if ((ros::Time::now() - hover_start_time) > hover_timeout) {
            // 已到达并稳定悬停1秒
            ROS_INFO("航点%d悬停完成，准备前往下一个航点", idx);
            return true;
        }
        return false; // 还未稳定悬停足够时间
    } else {
        // 只有距离大于阈值+0.1时才退出悬停
        if (distance > threshold + 0.1) {
            hovering = false;
            last_in_threshold = false;
        }
        
        // 高级PID控制 - 优化版本，提高响应速度和精度
        // 添加死区控制，避免微小误差导致的抖动
        float dead_zone = 0.05;  // 5cm死区
        
        float vx = 0.0, vy = 0.0, vz = 0.0;
        
        // X方向控制
        if (abs(dx) > dead_zone) {
            vx = limit(dx * current_pid_kp_x * current_speed_scale, -max_speed * current_speed_scale, max_speed * current_speed_scale);
        }
        
        // Y方向控制
        if (abs(dy) > dead_zone) {
            vy = limit(dy * current_pid_kp_y * current_speed_scale, -max_speed * current_speed_scale, max_speed * current_speed_scale);
        }
        
        // Z方向控制（更严格的高度控制）
        if (abs(dz) > dead_zone * 0.5) {  // Z轴死区更小
            vz = limit(dz * current_pid_kp_z * current_speed_scale * 0.8, -max_speed * current_speed_scale * 0.5, max_speed * current_speed_scale * 0.5);
        }

        current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
        current_goal.header.stamp = ros::Time::now();
        current_goal.type_mask = velocity_mask;
        current_goal.velocity.x = vx;
        current_goal.velocity.y = vy;
        current_goal.velocity.z = vz;
        current_goal.yaw = current_yaw;
        local_pos_pub.publish(current_goal);

        ROS_INFO("飞行中: 目标(%f, %f, %f), 当前位置(%.3f, %.3f, %.3f), 相对高度=%.3f, v=(%.2f, %.2f, %.2f), 距离=%.3f", 
                positions[idx].x(), positions[idx].y(), positions[idx].z(), 
                position_x, position_y, position_z, relative_z, vx, vy, vz, distance);
        return false; // 还未到达
    }
}


//校准对齐函数，，订阅/bboxes_pub，获取二维码中心点，计算与无人机中心点的偏差，发布校准指令
//订阅/bboxes_pub，获取二维码中心点 回调函数
void bboxes_cb(const yolov8_ros_msgs::BoundingBoxes::ConstPtr& msg)
{
    bboxes = *msg;
    
    // 检查是否有检测到的对象
    if (bboxes.bounding_boxes.empty()) {
        // 重要：没有检测到对象时，重置检测状态
        animal_detected = false;
        animal_confidence = 0.0;
        ROS_DEBUG("未检测到任何动物，重置检测状态");
        return;
    }
    
    // 遍历所有检测到的对象
    bool found_valid_animal = false;  // 标记是否找到有效的动物
    
    for (const auto& bbox : bboxes.bounding_boxes) {
        // 获取对象中心点坐标
        float center_x = bbox.center_x;  // 从Python代码中的tracker_out.center_x
        float center_y = bbox.center_y;  // 从Python代码中的tracker_out.center_y
        bool flag = bbox.flag;           // 从Python代码中的tracker_out.flag
        
        // 获取对象信息
        std::string object_class = bbox.Class;  // 对象类别
        float confidence = bbox.probability; // 置信度
        
        // 动物检测（只处理动物，不处理二维码）
        ROS_INFO("检测到动物: 类型=%s, 置信度=%.2f, 中心点=(%.1f, %.1f), flag=%d", 
                object_class.c_str(), confidence, center_x, center_y, flag);
        
        // 如果flag为true且置信度足够高，处理动物检测
        if (flag && confidence > 0.7) {
            // 更新动物检测全局变量
            animal_detected = true;
            detected_animal_type = object_class;
            animal_confidence = confidence;
            found_valid_animal = true;
            
            // 获取当前网格位置
            current_grid_position = world_to_grid_position(position_x, position_y);
            
            // 航点切换检测和状态重置
            if (current_grid_position != last_grid_position) {
                // 记录航点切换日志
                ROS_INFO("=== 航点切换日志 ===");
                ROS_INFO("时间: %.3f", ros::Time::now().toSec());
                ROS_INFO("位置切换: %s -> %s", last_grid_position.c_str(), current_grid_position.c_str());
                ROS_INFO("航点类型: %s", (duplicate_waypoints.find(current_grid_position) != duplicate_waypoints.end()) ? "重复航点" : "普通航点");
                ROS_INFO("检测状态: %s", animal_detected_at_duplicate ? "已检测" : "未检测");
                
                // 重置检测状态（按航点切换）
                animal_detected_at_duplicate = false;
                last_grid_position = current_grid_position;
                ROS_INFO("航点切换完成，重置检测状态");
            }
            
            // 检查是否在重复航点
            is_at_duplicate_waypoint = (duplicate_waypoints.find(current_grid_position) != duplicate_waypoints.end());
            
            // 记录动物检测日志
            ROS_INFO("=== 动物检测日志 ===");
            ROS_INFO("检测位置: %s", current_grid_position.c_str());
            ROS_INFO("动物类型: %s", detected_animal_type.c_str());
            ROS_INFO("置信度: %.2f", animal_confidence);
            
            if (is_at_duplicate_waypoint) {
                // 在重复航点，检查是否已经检测过
                if (!animal_detected_at_duplicate) {
                    // 第一次在重复航点检测到动物，发布位置信息并允许计数
                    publish_animal_location(current_grid_position, detected_animal_type);
                    animal_detected_at_duplicate = true;
                    ROS_INFO("计数决策: 允许计数");
                    ROS_INFO("决策原因: 重复航点首次检测");
                    // 发送计数允许信号给YOLO
                    std_msgs::Bool count_signal;
                    count_signal.data = true;
                    count_allow_pub.publish(count_signal);
                } else {
                    // 重复航点再次检测，禁止计数
                    ROS_INFO("计数决策: 禁止计数");
                    ROS_INFO("决策原因: 重复航点已检测过");
                    // 发送计数禁止信号给YOLO
                    std_msgs::Bool count_signal;
                    count_signal.data = false;
                    count_allow_pub.publish(count_signal);
                }
            } else {
                // 不在重复航点，正常发布和计数
                publish_animal_location(current_grid_position, detected_animal_type);
                ROS_INFO("计数决策: 允许计数");
                ROS_INFO("决策原因: 普通航点正常检测");
                // 发送计数允许信号给YOLO
                std_msgs::Bool count_signal;
                count_signal.data = true;
                count_allow_pub.publish(count_signal);
            }
        }
    }
    
    // 如果没有找到有效的动物，重置状态
    if (!found_valid_animal) {
        animal_detected = false;
        animal_confidence = 0.0;
    }
    
        ROS_INFO("收到检测结果: %zd个对象, animal_detected=%s",
             bboxes.bounding_boxes.size(), animal_detected ? "true" : "false");
}

// 像素坐标转世界坐标函数（考虑相机外参和无人机姿态）
void pixel_to_world_coordinates(float pixel_x, float pixel_y, float depth, 
                               float& world_x, float& world_y) {
    // 调用高级坐标转换函数，保持接口兼容性
    pixel_to_world_coordinates_advanced(pixel_x, pixel_y, depth, world_x, world_y, current_yaw);
}

// 高级坐标转换函数（使用TF2，考虑无人机姿态）
void pixel_to_world_coordinates_advanced(float pixel_x, float pixel_y, float depth, 
                                        float& world_x, float& world_y, float current_yaw) {
    // 相机内参（标定后）
    const double FX = 459.758142;  // 焦距x (像素)
    const double FY = 459.609932;  // 焦距y (像素)
    const double CX = 326.465004;  // 光心x (像素)
    const double CY = 233.811816;  // 光心y (像素)
    
    // 归一化坐标（相机坐标系）
    float normalized_x = (pixel_x - CX) / FX;
    float normalized_y = (pixel_y - CY) / FY;
    
    // 相机坐标系下的3D点
    float camera_x = normalized_x * depth;
    float camera_y = normalized_y * depth;
    // float camera_z = depth;  // 未使用，注释掉
    
    // 考虑相机外参偏移
    float body_x = camera_x + CAMERA_X;
    float body_y = camera_y + CAMERA_Y;
    
    // 考虑无人机偏航角旋转（将机体坐标系转换到世界坐标系）
// 坐标系定义：机头方向为X轴正方向，机头左边为Y轴正方向
    float cos_yaw = cos(current_yaw);
    float sin_yaw = sin(current_yaw);
    
    // 坐标转换：机头方向为X轴正方向
// body_x对应左右（Y轴），body_y对应前后（X轴）
world_x = body_y * cos_yaw - body_x * sin_yaw;  // X轴向前（机头方向）
world_y = body_y * sin_yaw + body_x * cos_yaw;  // Y轴向左（机头左边）
    
    ROS_DEBUG("高级坐标转换: 像素(%.1f, %.1f) -> 机体(%.3f, %.3f) -> 世界(%.3f, %.3f)m, 偏航=%.2f", 
              pixel_x, pixel_y, body_x, body_y, world_x, world_y, current_yaw);
}

// 二维码对齐控制函数（使用像素偏差直接控制）
// 动物检测对齐控制函数（已移除二维码相关功能）
void animal_alignment_control() {
    // 此函数已移除，因为我们现在只进行动物检测，不需要对齐功能
    ROS_DEBUG("动物检测对齐控制：此功能已移除");
}

// ========== 发布对齐后的中心坐标 ==========
void publish_aligned_center(float world_x, float world_y, float depth, int animal_id) {
    geometry_msgs::PoseStamped aligned_center_msg;
    
    // 设置消息头
    aligned_center_msg.header.stamp = ros::Time::now();
    aligned_center_msg.header.frame_id = "map";  // 使用map坐标系
    
    // 设置位置信息（世界坐标系）
    aligned_center_msg.pose.position.x = world_x;
    aligned_center_msg.pose.position.y = world_y;
    aligned_center_msg.pose.position.z = depth;
    
    // 设置姿态信息（四元数，这里设置为单位四元数表示无旋转）
    aligned_center_msg.pose.orientation.x = 0.0;
    aligned_center_msg.pose.orientation.y = 0.0;
    aligned_center_msg.pose.orientation.z = 0.0;
    aligned_center_msg.pose.orientation.w = 1.0;
    
    // 发布消息
    aligned_center_pub.publish(aligned_center_msg);
    
    ROS_INFO("发布动物中心坐标: 世界坐标(%.3f, %.3f, %.3f)m, 动物ID: %d", 
             world_x, world_y, depth, animal_id);
}

// 路径规划器回调函数
void planned_path_callback(const geometry_msgs::PoseArray::ConstPtr& msg) {
    dynamic_positions.clear();
    
    ROS_INFO("=== 收到动态路径规划消息 ===");
    ROS_INFO("消息帧ID: %s", msg->header.frame_id.c_str());
    ROS_INFO("消息时间戳: %.3f", msg->header.stamp.toSec());
    ROS_INFO("原始航点数量: %zd", msg->poses.size());
    
    // 路径验证：检查是否为空
    if (msg->poses.empty()) {
        ROS_WARN("收到空的动态路径，保持使用固定路径");
        use_dynamic_path = false;
        return;
    }
    
    // 放宽路径验证：检查航点数量是否合理（从5改为3）
    if (msg->poses.size() < 3) {
        ROS_WARN("动态路径航点数量过少(%zd)，可能计算错误，保持使用固定路径", msg->poses.size());
        use_dynamic_path = false;
        return;
    }
    
    // 放宽坐标范围验证
    bool valid_coordinates = true;
    for (size_t i = 0; i < msg->poses.size(); ++i) {
        const auto& pose = msg->poses[i];
        // 放宽坐标范围：X轴从-1.0~5.0改为-2.0~6.0，Y轴从-1.0~4.0改为-2.0~5.0，Z轴从0.5~2.0改为0.3~3.0
        if (pose.position.x < -2.0 || pose.position.x > 6.0 ||
            pose.position.y < -2.0 || pose.position.y > 5.0 ||
            pose.position.z < 0.3 || pose.position.z > 3.0) {
            ROS_WARN("航点%zd坐标超出合理范围: (%.2f, %.2f, %.2f)", 
                    i+1, pose.position.x, pose.position.y, pose.position.z);
            valid_coordinates = false;
        }
    }
    
    if (!valid_coordinates) {
        ROS_WARN("动态路径包含无效坐标，但继续使用动态路径（放宽验证）");
        // 不再因为坐标问题拒绝动态路径
    }
    
    // 加载有效路径
    for (size_t i = 0; i < msg->poses.size(); ++i) {
        const auto& pose = msg->poses[i];
        Eigen::Vector3d point(pose.position.x, pose.position.y, pose.position.z);
        dynamic_positions.push_back(point);
        
        // 只打印前5个和后5个航点的详细信息
        if (i < 5 || i >= msg->poses.size() - 5) {
            ROS_INFO("  航点%zd: X=%.2f(前), Y=%.2f(左), Z=%.2f(上)", 
                    i+1, point.x(), point.y(), point.z());
        } else if (i == 5) {
            ROS_INFO("  ... (省略中间航点) ...");
        }
    }
    
    use_dynamic_path = true;
    ROS_INFO("动态路径验证通过，已加载 %zd 个航点", dynamic_positions.size());
    ROS_INFO("下次任务执行时将使用动态路径");
    
    // 强制输出调试信息
    ROS_INFO("=== 动态路径设置确认 ===");
    ROS_INFO("use_dynamic_path = %s", use_dynamic_path ? "true" : "false");
    ROS_INFO("dynamic_positions.size() = %zd", dynamic_positions.size());
    ROS_INFO("前5个航点坐标:");
    for (size_t i = 0; i < std::min(size_t(5), dynamic_positions.size()); ++i) {
        ROS_INFO("  航点%zd: (%.2f, %.2f, %.2f)", i+1, 
                dynamic_positions[i].x(), dynamic_positions[i].y(), dynamic_positions[i].z());
    }
    ROS_INFO("=== 动态路径设置完成 ===");
    
    // 记录规划路径轨迹用于对比
    planned_path_trajectory = dynamic_positions;
    trajectory_recording_enabled = true;
    actual_flight_trajectory.clear();  // 清空之前的轨迹
    
    ROS_INFO("轨迹记录已启用，开始记录实际飞行轨迹");
    
    // 分析重复航点
    analyze_duplicate_waypoints(msg);
    
    ROS_INFO("=== 动态路径接收完成 ===");
}

// 返航路径回调函数
void return_path_callback(const geometry_msgs::PoseArray::ConstPtr& msg) {
    // 保持向后兼容性，但现在主要使用完整路径
    ROS_INFO("收到返航路径（兼容性），包含 %zd 个航点", msg->poses.size());
    
    // 清空返航路径
    return_path.clear();
    
    // 转换返航路径点
    for (const auto& pose : msg->poses) {
        Eigen::Vector3d waypoint(pose.position.x, pose.position.y, pose.position.z);
        return_path.push_back(waypoint);
    }
    
    // 标记已收到返航路径（兼容性）
    return_path_received = true;
    return_mission_started = false;
    return_waypoint_index = 0;
    
    ROS_INFO("返航路径已更新（兼容性），但主要使用完整路径执行");
}

// 规划统计信息回调函数
void planning_stats_callback(const std_msgs::String::ConstPtr& msg) {
    ROS_INFO("收到路径规划统计: %s", msg->data.c_str());
}

// 记录飞行轨迹点
void record_flight_trajectory(int waypoint_index, const std::string& grid_position) {
    if (!trajectory_recording_enabled) {
        return;
    }
    
    FlightPoint point;
    point.x = position_x;
    point.y = position_y;
    point.z = position_z;
    point.timestamp = ros::Time::now();
    point.waypoint_index = waypoint_index;
    point.grid_position = grid_position;
    
    actual_flight_trajectory.push_back(point);
    
    ROS_INFO("记录轨迹点 %d: (%.3f, %.3f, %.3f) - %s", 
             waypoint_index, point.x, point.y, point.z, grid_position.c_str());
}

// 分析轨迹精度
void analyze_trajectory_accuracy() {
    if (actual_flight_trajectory.empty() || planned_path_trajectory.empty()) {
        ROS_WARN("轨迹数据不足，无法分析精度");
        return;
    }
    
    // 计算轨迹精度统计
    double total_error = 0.0;
    int valid_points = 0;
    
    for (size_t i = 0; i < std::min(actual_flight_trajectory.size(), planned_path_trajectory.size()); ++i) {
        const auto& actual = actual_flight_trajectory[i];
        const auto& planned = planned_path_trajectory[i];
        
        double error = sqrt(
            pow(actual.x - planned.x(), 2) + 
            pow(actual.y - planned.y(), 2) + 
            pow(actual.z - planned.z(), 2)
        );
        total_error += error;
        valid_points++;
    }
    
    if (valid_points > 0) {
        double avg_error = total_error / valid_points;
        ROS_INFO("=== 轨迹精度分析 ===");
        ROS_INFO("有效对比点数: %d", valid_points);
        ROS_INFO("平均轨迹误差: %.3f米", avg_error);
        ROS_INFO("总轨迹误差: %.3f米", total_error);
        
        // 发布分析结果
        std_msgs::String analysis_msg;
        analysis_msg.data = "轨迹精度分析: 平均误差=" + std::to_string(avg_error) + "m, 总误差=" + std::to_string(total_error) + "m";
        trajectory_pub.publish(analysis_msg);
    }
}

void pid_params_callback(const std_msgs::String::ConstPtr& msg) {
    // 解析PID参数调整命令
    std::string command = msg->data;
    ROS_INFO("收到PID参数调整命令: %s", command.c_str());
    
    // 示例命令格式: "kp_x:0.2" 或 "speed_scale:0.5"
    if (command.find("kp_x:") == 0) {
        float new_kp = std::stof(command.substr(5));
        current_pid_kp_x = new_kp;
        ROS_INFO("调整X方向P增益: %.3f", current_pid_kp_x);
    } else if (command.find("kp_y:") == 0) {
        float new_kp = std::stof(command.substr(5));
        current_pid_kp_y = new_kp;
        ROS_INFO("调整Y方向P增益: %.3f", current_pid_kp_y);
    } else if (command.find("kp_z:") == 0) {
        float new_kp = std::stof(command.substr(5));
        current_pid_kp_z = new_kp;
        ROS_INFO("调整Z方向P增益: %.3f", current_pid_kp_z);
    } else if (command.find("speed_scale:") == 0) {
        float new_scale = std::stof(command.substr(12));
        current_speed_scale = new_scale;
        ROS_INFO("调整速度缩放因子: %.3f", current_speed_scale);
    } else if (command == "reset_pid") {
        // 重置为默认值
        current_pid_kp_x = PID_KP_X;
        current_pid_kp_y = PID_KP_Y;
        current_pid_kp_z = PID_KP_Z;
        current_speed_scale = SPEED_SCALE;
        ROS_INFO("重置PID参数为默认值");
    } else if (command == "show_pid") {
        // 显示当前PID参数
        ROS_INFO("=== 当前PID参数 ===");
        ROS_INFO("KP_X: %.3f", current_pid_kp_x);
        ROS_INFO("KP_Y: %.3f", current_pid_kp_y);
        ROS_INFO("KP_Z: %.3f", current_pid_kp_z);
        ROS_INFO("速度缩放: %.3f", current_speed_scale);
    } else {
        ROS_WARN("未知的PID参数调整命令: %s", command.c_str());
        ROS_INFO("可用命令: kp_x:值, kp_y:值, kp_z:值, speed_scale:值, reset_pid, show_pid");
    }
    
    // 参数验证和限制
    current_pid_kp_x = std::max(0.1f, std::min(2.0f, current_pid_kp_x));
    current_pid_kp_y = std::max(0.1f, std::min(2.0f, current_pid_kp_y));
    current_pid_kp_z = std::max(0.1f, std::min(2.0f, current_pid_kp_z));
    current_speed_scale = std::max(0.2f, std::min(1.5f, current_speed_scale));
    
    ROS_INFO("PID参数验证完成 - X:%.2f, Y:%.2f, Z:%.2f, Speed:%.2f", 
             current_pid_kp_x, current_pid_kp_y, current_pid_kp_z, current_speed_scale);
}

// 路径就绪回调函数
void path_ready_callback(const std_msgs::Bool::ConstPtr& msg) {
    if (msg->data) {
        ROS_INFO("✅ 路径规划器报告路径已就绪，可以开始任务执行");
    } else {
        ROS_INFO("路径规划器报告路径未就绪，等待路径规划...");
    }
}

// ========== 回调函数 ==========
void rc_cb(const mavros_msgs::RCIn::ConstPtr& msg)
{
  rc = *msg;
  rc_value = rc.channels[4];
}

void bat_cb(const sensor_msgs::BatteryStateConstPtr& msg)
{
    bat = *msg;
    rcv_stamp = ros::Time::now();
    double voltage = 0;
    for (size_t i = 0; i < bat.cell_voltage.size(); ++i) {
        voltage += bat.cell_voltage[i];
    }
    volt = 0.8 * volt + 0.2 * voltage;
    percentage = bat.percentage;
    
    if (percentage > 0.05) {
        if ((rcv_stamp - last_print_t).toSec() > 10) {
            ROS_INFO("[px4ctrl] Voltage=%.3f, percentage=%.3f", volt, percentage);
            last_print_t = rcv_stamp;
        }
    } else {
        if ((rcv_stamp - last_print_t).toSec() > 1) {
            last_print_t = rcv_stamp;
        }
    }
}

void state_cb(const mavros_msgs::State::ConstPtr& msg){
    current_state = *msg;
    if( current_state.mode == "OFFBOARD" ) {  
        ROS_INFO("检测到OFFBOARD模式");
        // 移除自动设置takeoff的逻辑，确保只有手动操作才能起飞
    }
}

void position_cb(const nav_msgs::Odometry::ConstPtr& msg)//
{
    position_msg = *msg;
    position_x = position_msg.pose.pose.position.x;
    position_y = position_msg.pose.pose.position.y;
    position_z = position_msg.pose.pose.position.z;

    // 更新速度信息
    odom_v.x() = position_msg.twist.twist.linear.x;
    odom_v.y() = position_msg.twist.twist.linear.y;
    odom_v.z() = position_msg.twist.twist.linear.z;
    
    // 检测异常下降
    static float last_position_z = position_z;
    static ros::Time last_check_time;
    static bool last_check_time_initialized = false;
    static int abnormal_drop_count = 0;  // 异常下降计数器
    static float max_drop_rate = 0.0;    // 最大下降速率
    
    if (!last_check_time_initialized) {
        last_check_time = ros::Time::now();
        last_check_time_initialized = true;
    }
    
    float time_diff = (ros::Time::now() - last_check_time).toSec();
    if (time_diff > 0.1) {  // 每0.1秒检查一次
        float z_change = position_z - last_position_z;
        float drop_rate = z_change / time_diff;  // 下降速率 (m/s)
        
        // 记录最大下降速率
        if (drop_rate < max_drop_rate) {
            max_drop_rate = drop_rate;
        }
        
        // 更严格的异常下降检测
        if (z_change < -0.2 && height_offset_set) {  // 降低阈值到0.2米
            abnormal_drop_count++;
            ROS_WARN("检测到快速下降！0.1秒内下降%.3f米，速率: %.2f m/s，计数: %d", 
                     -z_change, drop_rate, abnormal_drop_count);
            
            // 连续3次检测到异常下降才触发强制降落
            if (abnormal_drop_count >= 3) {
                ROS_ERROR("连续%d次检测到异常下降！最大下降速率: %.2f m/s，强制停止！", 
                         abnormal_drop_count, max_drop_rate);
                // 立即停止所有运动
                current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
                current_goal.header.stamp = ros::Time::now();
                current_goal.type_mask = velocity_mask;
                current_goal.velocity.x = 0.0;
                current_goal.velocity.y = 0.0;
                current_goal.velocity.z = 0.0;
                current_goal.yaw = current_yaw;
                local_pos_pub.publish(current_goal);
                state = LANDING;
                land = true;
            }
        } else {
            // 正常飞行时重置计数器
            if (abnormal_drop_count > 0) {
                ROS_INFO("高度变化正常，重置异常下降计数器");
                abnormal_drop_count = 0;
                max_drop_rate = 0.0;
            }
        }
        last_position_z = position_z;
        last_check_time = ros::Time::now();
    }

    // 高度零点补偿：优化后的稳定设置逻辑（考虑NED坐标系）
    static int stable_count = 0;
    static float last_z = 0.0;
    static ros::Time height_check_start_time;
    static bool height_check_started = false;
    
    if (!height_check_started) {
        height_check_start_time = ros::Time::now();
        height_check_started = true;
    }
    
    if (!height_offset_set) {
        float time_since_start = (ros::Time::now() - height_check_start_time).toSec();
        
        // 检查高度是否稳定（连续3次变化小于0.02米，或等待2秒后强制设置）
        if (std::abs(position_z - last_z) < 0.02) {
            stable_count++;
        } else {
            stable_count = 0;
        }
        last_z = position_z;
        
        // 条件1：连续3次稳定，或条件2：等待2秒后强制设置
        if (stable_count >= 3 || time_since_start > 2.0) {
            // 修正高度零点设置逻辑
            // 如果当前高度为负值，说明高度零点设置过高，需要调整
            if (position_z < 0) {
                // 如果高度为负，说明高度零点设置过高，重新设置为当前高度
                height_offset = position_z;
                ROS_WARN("检测到负高度值: %.3f，调整高度零点为当前高度", position_z);
            } else {
                // 正常情况，使用当前高度作为零点
                height_offset = position_z;
            }
            height_offset_set = true;
            ROS_INFO("高度零点已设置: 原始高度=%.3f, 稳定计数=%d, 等待时间=%.1fs", 
                     position_z, stable_count, time_since_start);
        } else {
            ROS_INFO_THROTTLE(0.5, "等待高度稳定... 当前: %.3f, 稳定计数: %d/3, 等待时间: %.1fs", 
                             position_z, stable_count, time_since_start);
        }
    }
    
    // 添加高度零点验证和修正
    if (height_offset_set) {
        float relative_z = position_z - height_offset;
        
        // 智能高度零点修正：检测并修正高度零点设置错误
        static int correction_count = 0;
        static ros::Time last_correction_time;
        static bool correction_time_initialized = false;
        
        if (!correction_time_initialized) {
            last_correction_time = ros::Time::now();
            correction_time_initialized = true;
        }
        
        // 如果相对高度持续为负，说明高度零点设置过高
        if (relative_z < -0.1) {  // 降低阈值，更敏感
            float time_since_correction = (ros::Time::now() - last_correction_time).toSec();
            
            // 每5秒最多修正一次，避免频繁修正
            if (time_since_correction > 5.0) {
                correction_count++;
                ROS_WARN("检测到高度零点设置过高！相对高度: %.3f，修正次数: %d", relative_z, correction_count);
                
                // 重新设置高度零点为当前高度
                height_offset = position_z;
                height_offset_set = true;
                last_correction_time = ros::Time::now();
                
                ROS_INFO("高度零点已自动修正: 新高度零点=%.3f", height_offset);
                
                // 如果修正次数过多，发出警告
                if (correction_count > 5) {
                    ROS_ERROR("高度零点修正次数过多！建议检查PX4传感器状态");
                    correction_count = 0;  // 重置计数器
                }
            }
        } else {
            // 高度正常时重置修正计数
            if (correction_count > 0) {
                ROS_INFO("高度显示正常，重置修正计数器");
                correction_count = 0;
            }
        }
    }

    tf2::Quaternion quat;
    tf2::convert(msg->pose.pose.orientation, quat);
    double roll, pitch, yaw;
    tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);
    current_yaw = yaw;
    
    // 全局高度安全检查 - 更智能的检测
    if (height_offset_set) {
        float relative_z = position_z - height_offset;
        static int low_height_count = 0;
        static ros::Time last_low_height_time;
        static bool low_height_time_initialized = false;
        
        if (!low_height_time_initialized) {
            last_low_height_time = ros::Time::now();
            low_height_time_initialized = true;
        }
        
        // 检查相对高度是否过低
        if (relative_z < -1.5) {  // 提高阈值到-1.5米
            low_height_count++;
            float time_since_low = (ros::Time::now() - last_low_height_time).toSec();
            
            ROS_WARN("检测到高度过低！相对高度: %.3f，持续时间: %.1f秒，计数: %d", 
                     relative_z, time_since_low, low_height_count);
            
            // 连续5秒高度过低或计数超过10次才触发强制降落
            if (time_since_low > 5.0 || low_height_count > 10) {
                ROS_ERROR("高度异常持续时间过长！相对高度: %.3f，持续时间: %.1f秒，强制降落", 
                         relative_z, time_since_low);
                state = LANDING;
                land = true;
                // 设置降落目标位置并强制发布智能降落指令
                set_landing_target_position();
                float vx, vy, vz;
                calculate_landing_velocity(vx, vy, vz);
                
                current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
                current_goal.header.stamp = ros::Time::now();
                current_goal.type_mask = velocity_mask;
                current_goal.velocity.x = vx;
                current_goal.velocity.y = vy;
                current_goal.velocity.z = vz;
                current_goal.yaw = current_yaw;
                local_pos_pub.publish(current_goal);
            }
        } else {
            // 高度正常时重置计数器
            if (low_height_count > 0) {
                ROS_INFO("高度恢复正常，重置高度异常计数器");
                low_height_count = 0;
                last_low_height_time = ros::Time::now();
            }
        }
    }
    
    // 添加状态监控
    ROS_INFO_THROTTLE(2.0, "状态监控: state=%d, mission_state=%d, startmission=%d, height_offset_set=%d, position=(%.2f,%.2f,%.2f)", 
                      state, mission_state, startmission, height_offset_set, position_x, position_y, position_z);
    
    // 添加高度零点调试信息
    if (height_offset_set) {
        float relative_z = position_z - height_offset;
        ROS_INFO_THROTTLE(5.0, "高度调试: 绝对高度=%.3f, 高度零点=%.3f, 相对高度=%.3f", 
                         position_z, height_offset, relative_z);
        
        // 添加高度显示问题检测
        static int negative_height_count = 0;
        if (relative_z < -0.1) {
            negative_height_count++;
            ROS_WARN_THROTTLE(2.0, "高度显示异常: 相对高度=%.3f (应为正值), 异常计数=%d", relative_z, negative_height_count);
            
            if (negative_height_count > 5) {
                ROS_ERROR("高度显示持续异常！无人机实际在上升但显示为下降，高度零点设置错误");
                negative_height_count = 0;  // 重置计数器
            }
        } else {
            if (negative_height_count > 0) {
                ROS_INFO("高度显示恢复正常");
                negative_height_count = 0;
            }
        }
    }
}

void twist_cb(const quadrotor_msgs::PositionCommand::ConstPtr& msg)
{
    // 只有在任务状态机处于MISSION状态时才设置receive为true
    if (mission_state == MISSION_HOVER || mission_state == MISSION_POINT1 || 
        mission_state == MISSION_POINT2) {
        receive = true;
        ROS_DEBUG("收到ego轨迹命令，receive设置为true");
    } else {
        ROS_DEBUG("收到ego轨迹命令，但当前不在任务执行状态，忽略receive设置");
    }
    
    target_time_ = ros::Time::now();
    ego = *msg;
    ego_pos_x = ego.position.x;
    ego_pos_y = ego.position.y;
    ego_pos_z = ego.position.z;
    ego_vel_x = ego.velocity.x;
    ego_vel_y = ego.velocity.y;
    ego_vel_z = ego.velocity.z;
    ego_a_x = ego.acceleration.x;
    ego_a_y = ego.acceleration.y;
    ego_a_z = ego.acceleration.z;
    ego_yaw = ego.yaw;
    ego_yaw_rate = ego.yaw_dot;
}

void takeoff_land_cb(const quadrotor_msgs::TakeoffLand::ConstPtr& msg)
{
    takeoffland = *msg;
    ROS_INFO("收到takeoff_land命令: %d", takeoffland.takeoff_land_cmd);
    
    // 命令优先级：强制降落 > 降落 > 起飞 > 任务启动
    if (takeoffland.takeoff_land_cmd == 4) {
        // 强制降落命令 - 最高优先级
        mission_state = MISSION_LAND;
        state = LANDING;
        land = true;
        takeoff = false;  // 清除起飞命令
        startmission = false;  // 清除任务启动命令
        ROS_WARN("收到强制降落命令！立即进入降落状态");
    } else if (takeoffland.takeoff_land_cmd == 2) {
        // 降落命令
        land = true;
        takeoff = false;  // 清除起飞命令
        startmission = false;  // 清除任务启动命令
        ROS_WARN("收到降落命令！当前状态: state=%d, mission_state=%d", state, mission_state);
    } else if (takeoffland.takeoff_land_cmd == 1) {
        // 起飞命令
        takeoff = true;
        startmission = false;  // 清除任务启动命令
        ROS_INFO("收到起飞命令");
    } else if (takeoffland.takeoff_land_cmd == 3) {
        // 任务启动命令
        startmission = true;
        takeoff = false;  // 清除起飞命令
        ROS_INFO("收到任务启动命令");
    } else {
        ROS_WARN("未知的takeoff_land命令: %d", takeoffland.takeoff_land_cmd);
    }
    
    ROS_INFO("takeoff_land命令处理完成: takeoff=%d, land=%d, startmission=%d", 
             takeoff, land, startmission);
}

// ========== 主控制循环 ==========
void cmdCallback(const ros::TimerEvent &e)
{
    // 紧急降落检查 - 最高优先级
    if (land) {
        ROS_ERROR("紧急降落命令！强制停止所有运动并降落！");
        state = LANDING;
        mission_state = MISSION_LAND;  // 同步任务状态机
        land = false;
        
        // 设置降落目标位置
        set_landing_target_position();
        
        // 计算智能降落速度
        float vx, vy, vz;
        calculate_landing_velocity(vx, vy, vz);
        
        // 发布智能降落指令
        current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
        current_goal.header.stamp = ros::Time::now();
        current_goal.type_mask = velocity_mask;
        current_goal.velocity.x = vx;
        current_goal.velocity.y = vy;
        current_goal.velocity.z = vz;
        current_goal.yaw = current_yaw;
        local_pos_pub.publish(current_goal);
        
        ROS_INFO("智能紧急降落启动: 目标位置(%.2f,%.2f), 当前速度(%.2f,%.2f,%.2f)", 
                 landing_target_position.x(), landing_target_position.y(), vx, vy, vz);
        return;  // 直接返回，不执行其他逻辑
    }
    
    // 添加状态转换日志
    static State_t last_state = INIT;
    if (state != last_state) {
        ROS_INFO("主状态机转换: %d -> %d", last_state, state);
        last_state = state;
    }
    
    switch (state) {
        case INIT: {
            if (current_state.mode == "OFFBOARD") {
                if (!current_state.armed && takeoff) {
                    ROS_INFO("收到手动起飞命令，开始解锁...");
                    if (arming_client.call(arm_cmd) && arm_cmd.response.success) {
                        ROS_INFO("解锁成功，开始起飞!!!");
                        state = TAKEOFF;
                        takeoff = false;  // 重置起飞标志
                    } else {
                        ROS_ERROR("解锁失败！");
                    }
                    last_request = ros::Time::now();
                } else if (takeoff) {
                    ROS_WARN("等待OFFBOARD模式... 当前模式: %s", current_state.mode.c_str());
                }
            } else {
                ROS_INFO_THROTTLE(2.0, "[control]INIT - 等待OFFBOARD模式，当前模式: %s", current_state.mode.c_str());
                last_request = ros::Time::now();
            }
            break;
        }
        case TAKEOFF: {
            // 等待高度零点设置完成
            if (!height_offset_set) {
                ROS_INFO_THROTTLE(1.0, "等待高度零点设置完成...");
                // 在零点设置前，保持当前位置
                current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
                current_goal.header.stamp = ros::Time::now();
                current_goal.type_mask = velocity_mask;
                current_goal.velocity.x = 0.0;
                current_goal.velocity.y = 0.0;
                current_goal.velocity.z = 0.0;
                current_goal.yaw = current_yaw;
                local_pos_pub.publish(current_goal);
                break;
            }
            
            // 添加高度安全检查
            float relative_z = position_z - height_offset;
            if (relative_z < -2.0) {
                ROS_ERROR("高度异常！相对高度: %.3f，强制降落", relative_z);
                state = LANDING;
                break;
            }
            
            // 优化后的起飞完成条件
            bool height_reached = relative_z >= TAKEOFF_HEIGHT * 0.85;  // 降低到85%
            bool speed_stable = odom_v.norm() < 0.3;  // 放宽速度要求
            bool position_stable = (std::abs(position_x) < 0.2) && (std::abs(position_y) < 0.2);  // 位置稳定
            
            if (height_reached && speed_stable && position_stable) {
                state = HOVER;
                targetpos_x = position_x;
                targetpos_y = position_y;
                ROS_INFO("起飞完成！高度: %.3f, 速度: %.3f, 位置: (%.2f,%.2f)", 
                         relative_z, odom_v.norm(), position_x, position_y);
            } else {
                // 优化的起飞控制：更平滑的速度控制
                float height_error = TAKEOFF_HEIGHT - relative_z;
                float vz_cmd = 0.0;
                
                if (height_error > 0.1f) {
                    // 使用更保守的P控制
                    vz_cmd = std::min(height_error * 0.3f, 0.5f);  // 最大速度限制为0.5m/s
                } else {
                    // 接近目标高度时，使用更小的速度
                    vz_cmd = std::min(height_error * 0.2f, 0.2f);
                }
                
                // 水平位置保持控制
                float vx_cmd = -position_x * 0.5;  // 简单的P控制回到原点
                float vy_cmd = -position_y * 0.5;
                
                // 限制水平速度
                float max_horizontal_speed = 0.3;
                float horizontal_speed = sqrt(vx_cmd * vx_cmd + vy_cmd * vy_cmd);
                if (horizontal_speed > max_horizontal_speed) {
                    vx_cmd = vx_cmd * max_horizontal_speed / horizontal_speed;
                    vy_cmd = vy_cmd * max_horizontal_speed / horizontal_speed;
                }
                
                current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
                current_goal.header.stamp = ros::Time::now();
                current_goal.type_mask = velocity_mask;
                current_goal.velocity.x = vx_cmd;
                current_goal.velocity.y = vy_cmd;
                current_goal.velocity.z = vz_cmd;
                current_goal.yaw = current_yaw;
                local_pos_pub.publish(current_goal);
                
                ROS_INFO_THROTTLE(1.0, "起飞中: 高度=%.3f/%.1f, 速度=%.3f, 位置=(%.2f,%.2f), 指令=(%.2f,%.2f,%.2f)", 
                                 relative_z, TAKEOFF_HEIGHT, odom_v.norm(), position_x, position_y, vx_cmd, vy_cmd, vz_cmd);
            }
            break;
        }
        case HOVER: {
            // 添加高度安全检查
            float relative_z = position_z - height_offset;
            if (relative_z < -2.0) {
                ROS_ERROR("悬停状态高度异常！相对高度: %.3f，强制降落", relative_z);
                state = LANDING;
                break;
            }
            
            // 当任务状态机在执行任务时，让出控制权
            if (mission_state == MISSION_POINT1 || mission_state == MISSION_HOVER_POINT || 
                mission_state == MISSION_QR_ALIGNMENT || mission_state == MISSION_SEARCH_QR || 
                mission_state == MISSION_HOVER_BEFORE_LAND) {
                // 任务状态机正在执行任务，主状态机不发布控制指令
                ROS_DEBUG_THROTTLE(1.0, "任务状态机在%d状态（网格路径飞行），主状态机让出控制权", mission_state);
                return; // 直接返回，不发布任何控制指令
            } else if (mission_state == MISSION_HOVER || mission_state == MISSION_POINT2) {
                if (receive) {
                    state = MISSION;
                    receive = false;
                    ROS_INFO("从HOVER状态进入MISSION状态，开始执行ego轨迹");
                } else {
                    // 如果没有ego轨迹，继续悬停等待任务状态机处理
                    current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
                    current_goal.header.stamp = ros::Time::now();
                    current_goal.type_mask = velocity_mask;
                    current_goal.velocity.x = (targetpos_x - position_x) * 0.5;  // 调小悬停速度
                    current_goal.velocity.y = (targetpos_y - position_y) * 0.5;  // 调小悬停速度
                    // 使用相对高度进行悬停控制，保持与Rough_flight逻辑一致
                    current_goal.velocity.z = (TAKEOFF_HEIGHT - relative_z) * 0.4;  // 调小悬停速度
                    current_goal.yaw = current_yaw;
                    local_pos_pub.publish(current_goal);
                }
            } else {
                // 非任务状态下的正常悬停
                current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
                current_goal.header.stamp = ros::Time::now();
                current_goal.type_mask = velocity_mask;
                current_goal.velocity.x = (targetpos_x - position_x) * 0.5;  // 调小悬停速度
                current_goal.velocity.y = (targetpos_y - position_y) * 0.5;  // 调小悬停速度
                // 使用相对高度进行悬停控制，保持与Rough_flight逻辑一致
                current_goal.velocity.z = (TAKEOFF_HEIGHT - relative_z) * 0.4;  // 调小悬停速度
                current_goal.yaw = current_yaw;
                local_pos_pub.publish(current_goal);
            }
            break;
        }
        case MISSION: {
            // 添加高度安全检查
            float relative_z = position_z - height_offset;
            if (relative_z < -2.0) {
                ROS_ERROR("任务状态高度异常！相对高度: %.3f，强制降落", relative_z);
                state = LANDING;
                break;
            }
            
            // 重置返航起飞点悬停状态（每次进入MISSION状态时）
            static bool mission_started = false;
            static State_t last_state = INIT;
            
            if (last_state != MISSION) {
                // 状态切换时重置
                mission_started = false;
                reset_return_takeoff_point_hover();
                ROS_INFO("�� 状态切换，重置返航起飞点悬停状态");
            }
            
            if (!mission_started) {
                mission_started = true;
                ROS_INFO("�� 任务开始，重置返航起飞点悬停状态");
            }
            
            last_state = MISSION;
            
            // 统一的路径执行逻辑（任务路径 + 返航路径）
            if (global_current_waypoint < static_cast<int>(dynamic_positions.size())) {
                // 还有航点需要执行
                Eigen::Vector3d target_waypoint = dynamic_positions[global_current_waypoint];
                
                // 计算到目标航点的距离
                float distance_to_target = sqrt(
                    pow(position_x - target_waypoint.x(), 2) + 
                    pow(position_y - target_waypoint.y(), 2) + 
                    pow(position_z - target_waypoint.z(), 2)
                );
                
                if (distance_to_target < 0.3) {
                    // 到达当前航点，移动到下一个
                    global_current_waypoint++;
                    
                    // 判断是否进入返航阶段
                    if (global_current_waypoint >= static_cast<int>(dynamic_positions.size()) - return_path.size()) {
                        ROS_INFO("✅ 任务完成，开始执行返航路径");
                    } else {
                        ROS_INFO("到达任务航点 %d/%zd，移动到下一个航点", 
                                global_current_waypoint, dynamic_positions.size());
                    }
                } else {
                    // 飞向当前航点
                    float vx = (target_waypoint.x() - position_x) * 0.5;
                    float vy = (target_waypoint.y() - position_y) * 0.5;
                    float vz = (target_waypoint.z() - position_z) * 0.3;
                    
                    // 限制速度
                    float max_speed = 0.5;
                    float speed = sqrt(vx*vx + vy*vy + vz*vz);
                    if (speed > max_speed) {
                        vx = vx * max_speed / speed;
                        vy = vy * max_speed / speed;
                        vz = vz * max_speed / speed;
                    }
                    
                    current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
                    current_goal.header.stamp = ros::Time::now();
                    current_goal.type_mask = velocity_mask;
                    current_goal.velocity.x = vx;
                    current_goal.velocity.y = vy;
                    current_goal.velocity.z = vz;
                    current_goal.yaw = current_yaw;
                    local_pos_pub.publish(current_goal);
                    
                    // 判断是否在返航阶段
                    bool is_return_phase = (global_current_waypoint >= static_cast<int>(dynamic_positions.size()) - return_path.size());
                    const char* phase_name = is_return_phase ? "返航" : "任务";
                    
                    ROS_INFO_THROTTLE(1.0, "%s中: 航点%d/%zd, 距离=%.2fm, 目标=(%.2f,%.2f,%.2f)", 
                                     phase_name, global_current_waypoint + 1, dynamic_positions.size(), distance_to_target,
                                     target_waypoint.x(), target_waypoint.y(), target_waypoint.z());
                }
            } else {
                // 所有航点执行完成，检查是否需要返航起飞点悬停
                if (return_takeoff_point_hover_enabled && !return_takeoff_point_hover_completed) {
                    if (is_at_return_takeoff_point()) {
                        // 到达返航起飞点A9B1(0,0)，开始悬停
                        if (!return_takeoff_point_hover_started) {
                            start_return_takeoff_point_hover();
                        }
                        
                        if (is_return_takeoff_point_hover_completed()) {
                            // 悬停完成，开始降落
                            ROS_INFO("✅ 返航起飞点悬停完成，开始降落");
                            set_landing_target_position();
                            state = LANDING;
                            land = true;
                        } else {
                            // 继续悬停
                            current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
                            current_goal.header.stamp = ros::Time::now();
                            current_goal.type_mask = velocity_mask;
                            current_goal.velocity.x = 0.0;
                            current_goal.velocity.y = 0.0;
                            current_goal.velocity.z = 0.0;
                            current_goal.yaw = current_yaw;
                            local_pos_pub.publish(current_goal);
                        }
                    } else {
                        // 还未到达返航起飞点，继续飞行
                        float vx = (return_takeoff_point.x() - position_x) * 0.5;
                        float vy = (return_takeoff_point.y() - position_y) * 0.5;
                        float vz = (return_takeoff_point.z() - position_z) * 0.3;
                        
                        // 限制速度
                        float max_speed = 0.5;
                        float speed = sqrt(vx*vx + vy*vy + vz*vz);
                        if (speed > max_speed) {
                            vx = vx * max_speed / speed;
                            vy = vy * max_speed / speed;
                            vz = vz * max_speed / speed;
                        }
                        
                        current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
                        current_goal.header.stamp = ros::Time::now();
                        current_goal.type_mask = velocity_mask;
                        current_goal.velocity.x = vx;
                        current_goal.velocity.y = vy;
                        current_goal.velocity.z = vz;
                        current_goal.yaw = current_yaw;
                        local_pos_pub.publish(current_goal);
                        
                        float distance_to_takeoff = sqrt(
                            pow(position_x - return_takeoff_point.x(), 2) + 
                            pow(position_y - return_takeoff_point.y(), 2) + 
                            pow(position_z - return_takeoff_point.z(), 2)
                        );
                        
                        ROS_INFO_THROTTLE(1.0, "飞向返航起飞点: 距离=%.2fm, 目标=(%.2f,%.2f,%.2f)", 
                                         distance_to_takeoff, return_takeoff_point.x(), return_takeoff_point.y(), return_takeoff_point.z());
                    }
                } else {
                    // 不启用悬停或悬停已完成，直接开始降落
                    ROS_INFO("✅ 完整路径执行完成，开始降落");
                    set_landing_target_position();
                    state = LANDING;
                    land = true;
                }
            }
            break;
        }

        //


        case LANDING: {
            ROS_INFO("降落状态: 当前高度=%.3f", position_z);
            if (position_z < 0.8) {
                state = AUTOLAND;
                mission_land_time = ros::Time::now();
                ROS_INFO("进入自动降落模式");
            } else {
                // 使用智能降落位置控制
                float vx, vy, vz;
                calculate_landing_velocity(vx, vy, vz);
                
                current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
                current_goal.header.stamp = ros::Time::now();
                current_goal.type_mask = velocity_mask;
                current_goal.velocity.x = vx;
                current_goal.velocity.y = vy;
                current_goal.velocity.z = vz;
                current_goal.yaw = current_yaw;
                local_pos_pub.publish(current_goal);
                
                ROS_INFO_THROTTLE(1.0, "智能降落中: 高度=%.3f, 速度=(%.2f,%.2f,%.2f), 目标位置(%.2f,%.2f)", 
                                 position_z, vx, vy, vz, landing_target_position.x(), landing_target_position.y());
            }
            break;
        }
        case AUTOLAND: {
            ros::Time time_now = ros::Time::now();
            if ((time_now - mission_land_time).toSec() < 10) {
                current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
                current_goal.header.stamp = ros::Time::now();
                current_goal.type_mask = velocity_mask;
                current_goal.velocity.x = 0;
                current_goal.velocity.y = 0;
                current_goal.velocity.z = -0.1;
                current_goal.yaw = current_yaw;
                local_pos_pub.publish(current_goal);
                ROS_INFO_THROTTLE(1.0, "自动降落中: 高度=%.3f, 下降速度=-0.1", position_z);
            } else {
                ROS_INFO("自动降落完成，重置状态");
                state = INIT;
                mission_state = MISSION_INIT;
            }
            break;
        }
    }
    
}

// ========== 任务调度 ==========
void missionCallback(const ros::TimerEvent &e)
{
    // 添加任务状态监控
    static Mission_State_t last_mission_state = MISSION_INIT;
    static ros::Time mission_start_time;
    static bool mission_start_time_initialized = false;
    
    if (!mission_start_time_initialized) {
        mission_start_time = ros::Time::now();
        mission_start_time_initialized = true;
    }
    
    if (mission_state != last_mission_state) {
        ROS_INFO("任务状态转换: %d -> %d", last_mission_state, mission_state);
        last_mission_state = mission_state;
        mission_start_time = ros::Time::now(); // 重置任务计时器
    }
    
    // 添加详细的状态信息输出
    ROS_INFO_THROTTLE(2.0, "任务状态机: state=%d, mission_state=%d, startmission=%d, receive=%d", 
                     state, mission_state, startmission, receive);
    
    // 全局高度安全检查 - 任务状态机版本
    if (height_offset_set) {
        float relative_z = position_z - height_offset;
        static int mission_low_height_count = 0;
        static ros::Time mission_last_low_height_time;
        static bool mission_low_height_time_initialized = false;
        
        if (!mission_low_height_time_initialized) {
            mission_last_low_height_time = ros::Time::now();
            mission_low_height_time_initialized = true;
        }
        
        // 检查相对高度是否过低
        if (relative_z < -1.2) {  // 任务状态机使用更严格的阈值
            mission_low_height_count++;
            float time_since_low = (ros::Time::now() - mission_last_low_height_time).toSec();
            
            ROS_WARN("任务状态机检测到高度过低！相对高度: %.3f，持续时间: %.1f秒，计数: %d", 
                     relative_z, time_since_low, mission_low_height_count);
            
            // 连续3秒高度过低或计数超过5次才触发强制降落
            if (time_since_low > 3.0 || mission_low_height_count > 5) {
                ROS_ERROR("任务状态机：高度异常持续时间过长！相对高度: %.3f，持续时间: %.1f秒，强制降落", 
                         relative_z, time_since_low);
                mission_state = MISSION_LAND;
                state = LANDING;
                return;
            }
        } else {
            // 高度正常时重置计数器
            if (mission_low_height_count > 0) {
                ROS_INFO("任务状态机：高度恢复正常，重置高度异常计数器");
                mission_low_height_count = 0;
                mission_last_low_height_time = ros::Time::now();
            }
        }
    }
    
    // 任务超时检查（如果任务运行超过5分钟，强制降落）
    if (mission_state != MISSION_INIT && (ros::Time::now() - mission_start_time).toSec() > 300.0) {
        ROS_WARN("任务超时！强制降落");
        mission_state = MISSION_LAND;
        state = LANDING;
        return;
    }
    
    // 如果主状态机处于降落状态，任务状态机应该同步
    if (state == LANDING || state == AUTOLAND) {
        mission_state = MISSION_LAND;
        return;
    }
    
    
    if (land) {
        ROS_ERROR("任务状态机检测到紧急降落命令！强制进入降落状态");
        mission_state = MISSION_LAND;
        state = LANDING;
        return;
    }
    
    switch (mission_state) {
        case MISSION_INIT: {
            // 任务启动条件
            if (startmission) {
                ROS_INFO("收到任务启动命令！");
                
                // 发布任务启动信号给路径规划器
                std_msgs::Bool mission_start_msg;
                mission_start_msg.data = true;
                mission_start_pub.publish(mission_start_msg);
                ROS_INFO("已发布任务启动信号给路径规划器");
                
                // 重置对齐完成标志
                alignment_completed = false;
                ROS_INFO("重置对齐完成标志，准备新的二维码对齐任务");
                
                // 重置全局航点变量
                global_current_waypoint = 0;
                global_waypoint_entered = false;
                global_waypoint_start_time = ros::Time::now();
                global_waypoint_started = false;
                global_first_enter = true;
                
                
                if (yaw_controller_active) {
                    stop_yaw_control();
                    ROS_INFO("重置转头控制器状态");
                }
                
                ROS_INFO("重置任务状态变量，准备新的任务执行");
                
                if (state == INIT) {
                    mission_state = MISSION_TAKEOFF;
                    takeoff = true;
                    ROS_INFO("从INIT状态启动任务: 起飞!");
                } else if (state == HOVER) {
                    mission_state = MISSION_HOVER;
                    mission_hover_time = ros::Time::now();
                    ROS_INFO("从HOVER状态启动任务: 准备前往目标点!");
                } else {
                    // 如果不在INIT或HOVER状态，等待状态稳定后再启动任务
                    ROS_WARN("当前状态不适合启动任务: state=%d, 等待状态稳定...", state);
                    
                    // 检查是否可以安全启动任务
                    if (state == TAKEOFF) {
                        ROS_INFO("无人机正在起飞，等待起飞完成后再启动任务");
                        // 不改变状态，等待起飞完成
                    } else if (state == MISSION) {
                        ROS_INFO("无人机正在执行任务，等待任务完成后再启动新任务");
                        // 不改变状态，等待任务完成
                    } else if (state == LANDING || state == AUTOLAND) {
                        ROS_WARN("无人机正在降落，无法启动新任务");
                        startmission = false;  // 重置标志位
                    } else {
                        // 其他未知状态，强制进入HOVER
                        ROS_WARN("未知状态，强制进入HOVER状态");
                        state = HOVER;
                        targetpos_x = position_x;
                        targetpos_y = position_y;
                        mission_state = MISSION_HOVER;
                        mission_hover_time = ros::Time::now();
                    }
                }
                startmission = false;  // 重置标志位
            } else {
                
                ROS_INFO_THROTTLE(3.0, "等待手动任务启动命令... startmission=%d, state=%d, mission_state=%d", 
                                 startmission, state, mission_state);
            }
            break;
        }
        case MISSION_TAKEOFF: {
            if (state == HOVER) {
                mission_state = MISSION_HOVER;
                mission_hover_time = ros::Time::now();
                ROS_INFO("起飞完成，准备前往目标点!");
            }
            break;
        }
        case MISSION_HOVER: {
            ros::Duration hover_elapsed = ros::Time::now() - mission_hover_time;
            if (hover_elapsed > ros::Duration(3.0)) {
                mission_state = MISSION_POINT1;
                
                // 检查动态路径是否可用
                if (use_dynamic_path && !dynamic_positions.empty()) {
                    ROS_INFO("悬停3秒完成，开始动态路径飞行，共%zd个航点", dynamic_positions.size());
                    ROS_INFO("动态路径已加载，将使用规划后的路径");
                } else {
                    ROS_INFO("悬停3秒完成，开始固定路径飞行，共%zd个航点", positions.size());
                    ROS_INFO("未检测到动态路径，使用固定路径");
                }
            } else {
                ROS_INFO_THROTTLE(1.0, "悬停中: 已悬停%.1f秒/3.0秒", hover_elapsed.toSec());
            }
            break;
        }
        
        case MISSION_POINT1: {
            // 使用全局变量而不是静态变量
            if (global_first_enter) {
                global_current_waypoint = 0;
                global_waypoint_entered = false;
                global_waypoint_start_time = ros::Time::now();
                global_waypoint_started = false;
                global_first_enter = false;
                ROS_INFO("首次进入MISSION_POINT1状态，重置所有航点变量");
            }
            
            // 选择使用的路径（增加安全检查）
            const std::vector<Eigen::Vector3d>& current_path = use_dynamic_path ? dynamic_positions : positions;
            const std::string path_type = use_dynamic_path ? "动态路径" : "固定路径";
            
            // 添加详细的路径状态调试信息
            ROS_INFO_THROTTLE(5.0, "=== 路径执行状态 ===");
            ROS_INFO_THROTTLE(5.0, "use_dynamic_path: %s", use_dynamic_path ? "true" : "false");
            ROS_INFO_THROTTLE(5.0, "dynamic_positions.size(): %zd", dynamic_positions.size());
            ROS_INFO_THROTTLE(5.0, "positions.size(): %zd", positions.size());
            ROS_INFO_THROTTLE(5.0, "当前使用路径: %s", path_type.c_str());
            ROS_INFO_THROTTLE(5.0, "当前路径航点数: %zd", current_path.size());
            ROS_INFO_THROTTLE(5.0, "当前航点索引: %d", global_current_waypoint);
            
            // 强制检查动态路径状态
            if (use_dynamic_path && dynamic_positions.empty()) {
                ROS_ERROR("use_dynamic_path为true但dynamic_positions为空！强制切换到固定路径");
                use_dynamic_path = false;
                return; // 等待下一个循环
            }
            
            if (!use_dynamic_path && !dynamic_positions.empty()) {
                ROS_WARN("检测到dynamic_positions有数据但use_dynamic_path为false，尝试启用动态路径");
                use_dynamic_path = true;
                ROS_INFO("已启用动态路径，航点数: %zd", dynamic_positions.size());
            }
            
            // 路径安全检查
            if (current_path.empty()) {
                ROS_ERROR("当前路径为空！强制使用固定路径");
                use_dynamic_path = false;
                // 重新选择路径
                const std::vector<Eigen::Vector3d>& safe_path = positions;
                ROS_INFO("已切换到固定路径，包含 %zd 个航点", safe_path.size());
                return; // 等待下一个循环
            }
            
            // 航点数量验证
            if (use_dynamic_path) {
                // 放宽航点数量要求：从50-70改为10-100
                if (current_path.size() < 10) {
                    ROS_WARN("动态路径航点数量过少(%zd)，可能计算错误，切换到固定路径", current_path.size());
                    use_dynamic_path = false;
                    return; // 等待下一个循环
                } else if (current_path.size() > 100) {
                    ROS_WARN("动态路径航点数量过多(%zd)，可能计算错误，切换到固定路径", current_path.size());
                    use_dynamic_path = false;
                    return; // 等待下一个循环
                } else {
                    ROS_INFO("动态路径航点数量正常: %zd个", current_path.size());
                }
            }
            
            // 航点索引安全检查
            if (global_current_waypoint >= static_cast<int>(current_path.size())) {
                ROS_ERROR("航点索引超出范围！当前索引=%d，路径大小=%zd，强制重置", 
                         global_current_waypoint, current_path.size());
                global_current_waypoint = 0;
                global_waypoint_entered = false;
                global_waypoint_started = false;
                global_first_enter = true;
                return;
            }
            
            if (!global_waypoint_entered) {
                ROS_INFO("=== 进入网格路径飞行状态 ===");
                ROS_INFO("路径选择: %s", path_type.c_str());
                if (use_dynamic_path) {
                    ROS_INFO("动态路径来源: /planned_path 话题");
                    ROS_INFO("动态路径接收时间: 任务开始前");
                } else {
                    ROS_INFO("固定路径来源: 代码中硬编码的positions数组");
                }
                ROS_INFO("总航点数: %zd", current_path.size());
                ROS_INFO("当前航点: %d/%zd", global_current_waypoint + 1, current_path.size());
                ROS_INFO("目标点: X=%.1f(前), Y=%.1f(左), Z=%.1f(上)", 
                        current_path[global_current_waypoint].x(), current_path[global_current_waypoint].y(), current_path[global_current_waypoint].z());
                ROS_INFO("当前位置: X=%.3f(前), Y=%.3f(左), Z=%.3f(上)", position_x, position_y, position_z);
                global_waypoint_entered = true;
                
                // 重置超时计时器
                global_waypoint_start_time = ros::Time::now();
                global_waypoint_started = true;
                ROS_INFO("开始前往航点%d，启动超时计时器", global_current_waypoint + 1);
            }
            
            // 使用粗略飞行控制到达当前目标点
            if (Rough_flight(current_path, global_current_waypoint, THRESHOLD, KP, MAX_SPEED)) {
                // 到达当前航点
                ROS_INFO("航点%d到达完成!", global_current_waypoint + 1);
                
                // 记录飞行轨迹
                std::string grid_position = "航点" + std::to_string(global_current_waypoint + 1);
                record_flight_trajectory(global_current_waypoint, grid_position);
                
                global_current_waypoint++;
                
                if (global_current_waypoint >= static_cast<int>(current_path.size())) {
                    // 所有航点完成，进入降落状态
                    ROS_INFO("所有航点完成！共飞行了%zd个航点", current_path.size());
                    
                    // 分析轨迹精度
                    if (trajectory_recording_enabled) {
                        analyze_trajectory_accuracy();
                        trajectory_recording_enabled = false;  // 停止记录
                    }
                    
                    mission_state = MISSION_LAND;
                    global_current_waypoint = 0; // 重置航点索引
                    global_waypoint_entered = false;
                    global_waypoint_started = false;
                    global_first_enter = true;  // 重置首次进入标志，为下次任务做准备
                    use_dynamic_path = false; // 重置动态路径标志
                } else {
                    // 继续下一个航点
                    global_waypoint_entered = false; // 重置标志，为下一个航点做准备
                    ROS_INFO("准备前往下一个航点: %d/%zd", global_current_waypoint + 1, current_path.size());
                }
            } else {
                // 航点任务超时检查（如果超过30秒未到达目标点，跳过该航点）
                if ((ros::Time::now() - global_waypoint_start_time).toSec() > 30.0) {
                    ROS_WARN("航点%d超时！跳过该航点", global_current_waypoint + 1);
                    global_current_waypoint++;
                    global_waypoint_entered = false; // 重置标志
                    global_waypoint_started = false; // 重置开始标志
                    
                    if (global_current_waypoint >= static_cast<int>(current_path.size())) {
                        // 所有航点完成，进入降落状态
                        ROS_INFO("所有航点完成！共飞行了%zd个航点", current_path.size());
                        mission_state = MISSION_LAND;
                        global_current_waypoint = 0; // 重置航点索引
                        global_waypoint_started = false;
                        global_first_enter = true;  // 重置首次进入标志，为下次任务做准备
                        use_dynamic_path = false; // 重置动态路径标志
                    } else {
                        // 还有更多航点，继续执行
                        ROS_INFO("跳过超时航点，继续前往下一个航点: %d/%zd", global_current_waypoint + 1, current_path.size());
                        // 重置超时计时器，为下一个航点做准备
                        global_waypoint_start_time = ros::Time::now();
                    }
                } else {
                    ROS_INFO_THROTTLE(2.0, "前往航点%d中: 已飞行%.1f秒/30.0秒", 
                                     global_current_waypoint + 1, (ros::Time::now() - global_waypoint_start_time).toSec());
                }
            }
            break;
        }
        
        case MISSION_HOVER_POINT: {
            // 判断悬停时间是否到
            if (ros::Time::now() - mission_hover_time > ros::Duration(10.0)) {
                mission_state = MISSION_LAND;
                ROS_INFO("悬停5秒完成，准备降落!");
            } else {
                // 持续发布悬停指令
                current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
                current_goal.header.stamp = ros::Time::now();
                current_goal.type_mask = velocity_mask;
                current_goal.velocity.x = 0.0;
                current_goal.velocity.y = 0.0;
                current_goal.velocity.z = 0.0;
                current_goal.yaw = current_yaw;
                local_pos_pub.publish(current_goal);   
            }
            break;
        }

        case MISSION_LAND: {
            // 检查是否需要返航起飞点悬停
            if (return_takeoff_point_hover_enabled && !return_takeoff_point_hover_completed) {
                if (is_at_return_takeoff_point()) {
                    // 到达返航起飞点A9B1(0,0)，开始悬停
                    if (!return_takeoff_point_hover_started) {
                        start_return_takeoff_point_hover();
                    }
                    
                    if (is_return_takeoff_point_hover_completed()) {
                        // 悬停完成，开始降落
                        ROS_INFO("✅ 返航起飞点悬停完成，开始降落");
                        set_landing_target_position();
                        state = LANDING;
                        land = true;
                    } else {
                        // 继续悬停
                        current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
                        current_goal.header.stamp = ros::Time::now();
                        current_goal.type_mask = velocity_mask;
                        current_goal.velocity.x = 0.0;
                        current_goal.velocity.y = 0.0;
                        current_goal.velocity.z = 0.0;
                        current_goal.yaw = current_yaw;
                        local_pos_pub.publish(current_goal);
                    }
                } else {
                    // 还未到达返航起飞点，继续飞行
                    float vx = (return_takeoff_point.x() - position_x) * 0.5;
                    float vy = (return_takeoff_point.y() - position_y) * 0.5;
                    float vz = (return_takeoff_point.z() - position_z) * 0.3;
                    
                    // 限制速度
                    float max_speed = 0.5;
                    float speed = sqrt(vx*vx + vy*vy + vz*vz);
                    if (speed > max_speed) {
                        vx = vx * max_speed / speed;
                        vy = vy * max_speed / speed;
                        vz = vz * max_speed / speed;
                    }
                    
                    current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
                    current_goal.header.stamp = ros::Time::now();
                    current_goal.type_mask = velocity_mask;
                    current_goal.velocity.x = vx;
                    current_goal.velocity.y = vy;
                    current_goal.velocity.z = vz;
                    current_goal.yaw = current_yaw;
                    local_pos_pub.publish(current_goal);
                    
                    float distance_to_takeoff = sqrt(
                        pow(position_x - return_takeoff_point.x(), 2) + 
                        pow(position_y - return_takeoff_point.y(), 2) + 
                        pow(position_z - return_takeoff_point.z(), 2)
                    );
                    
                    ROS_INFO_THROTTLE(1.0, "MISSION_LAND: 飞向返航起飞点: 距离=%.2fm, 目标=(%.2f,%.2f,%.2f)", 
                                     distance_to_takeoff, return_takeoff_point.x(), return_takeoff_point.y(), return_takeoff_point.z());
                }
            } else {
                // 不启用悬停或悬停已完成，直接开始降落
                if (state != LANDING && state != AUTOLAND) {
                    set_landing_target_position();
                    state = LANDING;
                    land = true;
                    ROS_INFO("任务完成，开始飞回起飞点A9B1进行降落");
                }
            }
            break;
        }
        default: {
            break;
        }
    }
}

// ========== 主函数 ==========
int main(int argc, char **argv)
{
    ros::init(argc, argv, "px4ctrl");
    setlocale(LC_ALL, "");
    ros::NodeHandle nh;
    
    // 初始化全局变量
    global_waypoint_start_time = ros::Time::now();

    // 初始化ROS订阅者和发布者
    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>("/mavros/state", 100, state_cb);
    local_pos_pub = nh.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 1);
    ros::Subscriber rc_sub = nh.subscribe<mavros_msgs::RCIn>("/mavros/rc/in", 10, rc_cb);
    ros::Subscriber bat_sub = nh.subscribe<sensor_msgs::BatteryState>("/mavros/battery", 10, bat_cb);
    arming_client = nh.serviceClient<mavros_msgs::CommandBool>("/mavros/cmd/arming");
    command_client = nh.serviceClient<mavros_msgs::CommandLong>("/mavros/cmd/command");
    set_mode_client = nh.serviceClient<mavros_msgs::SetMode>("/mavros/set_mode");
    ros::Subscriber twist_sub = nh.subscribe<quadrotor_msgs::PositionCommand>("/position_cmd", 10, twist_cb);
    ros::Subscriber takeoff_land_sub = nh.subscribe<quadrotor_msgs::TakeoffLand>("/takeoff_land", 10, takeoff_land_cb);
    pos_cmd_pub = nh.advertise<quadrotor_msgs::GoalSet>("/goal", 50);
    ros::Subscriber position_sub = nh.subscribe<nav_msgs::Odometry>("/mavros/local_position/odom", 10, position_cb);
    //订阅二维码中心点
    ros::Subscriber bboxes_sub = nh.subscribe<yolov8_ros_msgs::BoundingBoxes>("/bboxes_pub", 10, bboxes_cb);
    
    // 订阅动物统计信息
    ros::Subscriber animal_stats_sub = nh.subscribe<std_msgs::String>("/animal_statistics", 10, animal_stats_callback);
    
    // 发布对齐后的中心坐标
    aligned_center_pub = nh.advertise<geometry_msgs::PoseStamped>("/aligned_center", 10);
    // 订阅完整路径（任务+返航，主控制器执行用）
    ros::Subscriber complete_path_sub = nh.subscribe<geometry_msgs::PoseArray>("/complete_path", 10, planned_path_callback);
    
    // 订阅路径规划器发布的返航路径
    ros::Subscriber return_path_sub = nh.subscribe<geometry_msgs::PoseArray>("/return_path", 10, return_path_callback);
    
    // 订阅路径就绪信号
    ros::Subscriber path_ready_sub = nh.subscribe<std_msgs::Bool>("/path_ready", 10, path_ready_callback);
    
    // 发布任务启动信号
    mission_start_pub = nh.advertise<std_msgs::Bool>("/mission_start_signal", 10);
    
    // 发布动物位置信息
    animal_location_pub = nh.advertise<std_msgs::String>("/animal_location", 10);
    
    // 发布计数允许信号
    count_allow_pub = nh.advertise<std_msgs::Bool>("/count_allow", 10);
    
    // 轨迹记录与对比相关
    trajectory_pub = nh.advertise<std_msgs::String>("/flight_trajectory_analysis", 10);
    planning_stats_sub = nh.subscribe<std_msgs::String>("/path_planning_stats", 10, planning_stats_callback);
    
    // PID参数调整话题
    pid_params_sub = nh.subscribe<std_msgs::String>("/pid_params", 10, pid_params_callback);
    
    // 初始化TF2坐标转换
    tf_listener = new tf2_ros::TransformListener(tf_buffer);
    tf_available = true;
    ROS_INFO("TF2坐标转换器已初始化");
    
    // 初始化服务
    offb_set_mode.request.custom_mode = "OFFBOARD";
    arm_cmd.request.value = true;

    
    // 等待连接
    ros::Rate rate(50.0);
    last_print_t = ros::Time::now();
    while (ros::ok() && !current_state.connected) {
        ros::spinOnce();
        rate.sleep();
    }

    // 发送初始设置点
    for (int i = 100; ros::ok() && i > 0; --i) {
        current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
        local_pos_pub.publish(current_goal);
        ros::spinOnce();
        rate.sleep();
    }
    
    // 初始化状态
    last_request = ros::Time::now();
    state = INIT;
    mission_state = MISSION_INIT;
    
    // 创建定时器
    ros::Timer cmd_timer = nh.createTimer(ros::Duration(0.02), cmdCallback);
    ros::Timer mission_timer = nh.createTimer(ros::Duration(0.1), missionCallback);
    
    ROS_INFO("[px4ctrl]: 准备就绪，等待任务启动...");
    ROS_INFO("[px4ctrl]: 坐标系配置: 机头方向=X轴正方向(向前), Y轴正方向(向左), Z轴正方向(向上)");
    ROS_INFO("[px4ctrl]: 固定航点配置:");
    for (size_t i = 0; i < positions.size(); ++i) {
                    ROS_INFO("  航点%zd: X=%.1f(前), Y=%.1f(左), Z=%.1f(上)", i, positions[i].x(), positions[i].y(), positions[i].z());
    }
    ROS_INFO("[px4ctrl]: 支持动态路径规划，订阅话题: /complete_path（主控制器执行）");
    ROS_INFO("[px4ctrl]: 禁飞区域话题: /no_fly_zones");
    ROS_INFO("[px4ctrl]: PID参数调整话题: /pid_params");
    ROS_INFO("[px4ctrl]: 动物统计信息话题: /animal_statistics");
    ROS_INFO("[px4ctrl]: 当前PID参数 - KP_X:%.3f, KP_Y:%.3f, KP_Z:%.3f, 速度缩放:%.3f", 
             current_pid_kp_x, current_pid_kp_y, current_pid_kp_z, current_speed_scale);
    ROS_INFO("[px4ctrl]: 航点悬停时间: %.1f秒", HOVER_TIME);
    
    // 移除自动启动功能，确保只有手动操作才能起飞
    ROS_INFO("[px4ctrl]: 等待手动任务启动命令...");
    ROS_INFO("[px4ctrl]: 请通过遥控器或发送takeoff_land命令来启动任务");
    
    ros::spin();

    // 清理TF2资源
    if (tf_listener != nullptr) {
        delete tf_listener;
        tf_listener = nullptr;
    }

    return 0;
}

// ========== 坐标转换和重复航点检测函数实现 ==========

std::string world_to_grid_position(float x, float y) {
    // 世界坐标转换为网格坐标（A1B1格式）
    // 网格参数：0.5米一格
    // 坐标系：X轴向前（机头方向），Y轴向左（机头左边）
    // 路径规划器坐标系统：
    // x = (9-1-col) * 0.5 = (8-col) * 0.5
    // y = row * 0.5
    // 所以：col = 8 - x/0.5, row = y/0.5
    
    // 根据路径规划器的坐标系统计算网格索引
    int col = static_cast<int>(8 - x / 0.5) + 1;  // A列（1-9）
    int row = static_cast<int>(y / 0.5) + 1;      // B列（1-7）
    
    // 边界检查
    if (col < 1) col = 1;
    if (col > 9) col = 9;
    if (row < 1) row = 1;
    if (row > 7) row = 7;
    
    char grid_pos[10];
    snprintf(grid_pos, sizeof(grid_pos), "A%dB%d", col, row);
    return std::string(grid_pos);
}

void analyze_duplicate_waypoints(const geometry_msgs::PoseArray::ConstPtr& path_msg) {
    // 分析路径中的重复航点
    waypoint_visit_count.clear();
    duplicate_waypoints.clear();
    
    ROS_INFO("开始分析路径中的重复航点...");
    
    for (const auto& pose : path_msg->poses) {
        std::string grid_pos = world_to_grid_position(pose.position.x, pose.position.y);
        waypoint_visit_count[grid_pos]++;
        
        if (waypoint_visit_count[grid_pos] > 1) {
            duplicate_waypoints.insert(grid_pos);
            ROS_INFO("发现重复航点: %s (访问次数: %d)", grid_pos.c_str(), waypoint_visit_count[grid_pos]);
        }
    }
    
    ROS_INFO("重复航点分析完成，共发现 %zu 个重复航点", duplicate_waypoints.size());
    for (const auto& pos : duplicate_waypoints) {
        ROS_INFO("  - %s (访问次数: %d)", pos.c_str(), waypoint_visit_count[pos]);
    }
}

void publish_animal_location(const std::string& grid_pos, const std::string& animal_type) {
    // 发布动物位置信息到地面站
    std_msgs::String animal_msg;
    animal_msg.data = grid_pos + "," + animal_type;
    animal_location_pub.publish(animal_msg);
    
    ROS_INFO("发布动物位置信息: %s 在 %s", animal_type.c_str(), grid_pos.c_str());
}

void animal_stats_callback(const std_msgs::String::ConstPtr& msg) {
    // 解析YOLO发布的动物统计信息
    try {
        std::string json_str = msg->data;
        ROS_INFO("收到动物统计信息: %s", json_str.c_str());
        
        // 简单的JSON解析（不使用外部库）
        // 解析格式：{"timestamp": 1705123456.789, "total_count": 25, "animal_counts": {...}, ...}
        
        // 解析总数量
        size_t total_count_pos = json_str.find("\"total_count\":");
        if (total_count_pos != std::string::npos) {
            size_t start = json_str.find(":", total_count_pos) + 1;
            size_t end = json_str.find(",", start);
            if (end == std::string::npos) end = json_str.find("}", start);
            if (end != std::string::npos) {
                std::string total_count_str = json_str.substr(start, end - start);
                animal_stats.total_count = std::stoi(total_count_str);
            }
        }
        
        // 解析动物计数
        size_t animal_counts_pos = json_str.find("\"animal_counts\":");
        if (animal_counts_pos != std::string::npos) {
            size_t start = json_str.find("{", animal_counts_pos);
            size_t end = json_str.find("}", start);
            if (start != std::string::npos && end != std::string::npos) {
                std::string counts_str = json_str.substr(start + 1, end - start - 1);
                parse_animal_counts(counts_str, animal_stats.animal_counts);
            }
        }
        
        // 解析当前帧计数
        size_t current_frame_pos = json_str.find("\"current_frame_counts\":");
        if (current_frame_pos != std::string::npos) {
            size_t start = json_str.find("{", current_frame_pos);
            size_t end = json_str.find("}", start);
            if (start != std::string::npos && end != std::string::npos) {
                std::string counts_str = json_str.substr(start + 1, end - start - 1);
                parse_animal_counts(counts_str, animal_stats.current_frame_counts);
            }
        }
        
        // 解析检测时间
        size_t time_pos = json_str.find("\"detection_time\":");
        if (time_pos != std::string::npos) {
            size_t start = json_str.find("\"", time_pos + 16) + 1;
            size_t end = json_str.find("\"", start);
            if (start != std::string::npos && end != std::string::npos) {
                animal_stats.detection_time = json_str.substr(start, end - start);
            }
        }
        
        // 标记数据有效
        animal_stats.data_valid = true;
        
        // 打印解析结果
        ROS_INFO("解析结果 - 总数: %d, 时间: %s", 
                 animal_stats.total_count, animal_stats.detection_time.c_str());
        
        // 打印动物计数
        for (const auto& pair : animal_stats.animal_counts) {
            ROS_INFO("  %s: %d", pair.first.c_str(), pair.second);
        }
        
        // 主控决策逻辑
        make_flight_decisions();
        
    } catch (const std::exception& e) {
        ROS_WARN("解析动物统计信息失败: %s", e.what());
        animal_stats.data_valid = false;
    }
}

void parse_animal_counts(const std::string& counts_str, std::map<std::string, int>& counts_map) {
    // 解析动物计数字符串，格式: "Wolf": 5, "Elephant": 8, ...
    counts_map.clear();
    
    size_t pos = 0;
    while (pos < counts_str.length()) {
        // 查找动物名称
        size_t name_start = counts_str.find("\"", pos);
        if (name_start == std::string::npos) break;
        
        size_t name_end = counts_str.find("\"", name_start + 1);
        if (name_end == std::string::npos) break;
        
        std::string animal_name = counts_str.substr(name_start + 1, name_end - name_start - 1);
        
        // 查找数量
        size_t count_start = counts_str.find(":", name_end) + 1;
        if (count_start == std::string::npos) break;
        
        size_t count_end = counts_str.find(",", count_start);
        if (count_end == std::string::npos) count_end = counts_str.find("}", count_start);
        if (count_end == std::string::npos) break;
        
        std::string count_str = counts_str.substr(count_start, count_end - count_start);
        // 去除空格
        count_str.erase(0, count_str.find_first_not_of(" \t"));
        count_str.erase(count_str.find_last_not_of(" \t") + 1);
        
        try {
            int count = std::stoi(count_str);
            counts_map[animal_name] = count;
        } catch (...) {
            ROS_WARN("解析动物数量失败: %s", count_str.c_str());
        }
        
        pos = count_end + 1;
    }
}

void make_flight_decisions() {
    // 基于动物统计信息进行飞行决策
    
    if (!animal_stats.data_valid) {
        ROS_WARN("动物统计数据无效，无法进行决策");
        return;
    }
    
    // 决策1: 根据总检测数量判断任务进度
    if (animal_stats.total_count >= 50) {
        ROS_INFO("检测数量充足(%d)，考虑结束任务", animal_stats.total_count);
        // 这里可以设置任务完成标志
    } else if (animal_stats.total_count >= 30) {
        ROS_INFO("检测数量良好(%d)，继续任务", animal_stats.total_count);
    } else {
        ROS_INFO("检测数量较少(%d)，需要更多检测", animal_stats.total_count);
    }
    
    // 决策2: 根据当前帧检测情况调整策略
    int current_frame_total = 0;
    for (const auto& pair : animal_stats.current_frame_counts) {
        current_frame_total += pair.second;
    }
    
    if (current_frame_total > 0) {
        ROS_INFO("当前帧检测到 %d 个动物，保持当前飞行策略", current_frame_total);
    } else {
        ROS_INFO("当前帧无检测，可能需要调整飞行路径");
    }
    
    // 决策3: 检查检测系统健康状态
    static ros::Time last_valid_time = ros::Time::now();
    if (animal_stats.data_valid) {
        last_valid_time = ros::Time::now();
    } else {
        ros::Duration time_since_last = ros::Time::now() - last_valid_time;
        if (time_since_last.toSec() > 10.0) {
            ROS_WARN("检测系统可能异常，超过10秒未收到有效数据");
        }
    }
}

