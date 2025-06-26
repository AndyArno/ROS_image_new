#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf2_ros
import numpy as np
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import LaserScan

class RefactoredPointTransformer:
    def __init__(self):
        """
        重构后的坐标变换与融合节点
        - 从ROS参数服务器加载配置
        - 增加TF等待机制，提高稳健性
        """
        rospy.init_node('transformer_node', anonymous=False)

        self.load_params()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.lidar_data = None
        self.last_scale = None

        # 订阅与发布
        self.point_sub = rospy.Subscriber(self.input_point_topic, PointStamped, self.point_callback)
        self.lidar_sub = rospy.Subscriber(self.lidar_topic, LaserScan, self.lidar_callback)
        self.base_point_pub = rospy.Publisher(self.output_base_point_topic, PointStamped, queue_size=10)

        # 等待必要的TF变换关系建立
        self.wait_for_transforms()
        rospy.loginfo("坐标变换节点启动成功。")

    def load_params(self):
        """从ROS参数服务器加载所有配置"""
        try:
            self.lidar_topic = rospy.get_param('~lidar_topic')
            self.input_point_topic = rospy.get_param('~input_point_topic')
            self.output_base_point_topic = rospy.get_param('~output_base_point_topic')
            self.base_frame = rospy.get_param('~base_frame')
            self.lidar_frame = rospy.get_param('~lidar_frame')
            self.camera_optical_frame = rospy.get_param('~camera_optical_frame')
            self.angle_comp = rospy.get_param('~angle_compensation')
            self.special_idx = rospy.get_param('~special_angle_index')
            self.special_dist = rospy.get_param('~special_distance')
        except KeyError as e:
            rospy.logerr(f"参数加载失败，请检查launch文件和yaml文件: {e}")
            rospy.signal_shutdown("缺少关键参数")

    def wait_for_transforms(self):
        """启动时等待关键TF变换就绪，增加程序稳健性"""
        rospy.loginfo("等待TF变换关系...")
        try:
            # 等待相机到雷达的变换
            self.tf_buffer.can_transform(self.lidar_frame, self.camera_optical_frame, rospy.Time(0), rospy.Duration(10.0))
            # 等待雷达到底盘的变换
            self.tf_buffer.can_transform(self.base_frame, self.lidar_frame, rospy.Time(0), rospy.Duration(10.0))
            rospy.loginfo("TF变换关系已就绪。")
        except tf2_ros.TransformException as e:
            rospy.logerr(f"获取TF变换超时: {e}")
            rospy.signal_shutdown("无法建立TF树")

    def lidar_callback(self, msg):
        self.lidar_data = msg

    def point_callback(self, msg):
        if self.lidar_data is None:
            rospy.logwarn_throttle(2, "尚未接收到激光雷达数据，跳过处理。")
            return

        try:
            transform_cam_to_laser = self.tf_buffer.lookup_transform(
                self.lidar_frame, msg.header.frame_id, rospy.Time(0), rospy.Duration(0.1)
            )

            point_in_laser = tf2_geometry_msgs.do_transform_point(msg, transform_cam_to_laser)

            angle = np.arctan2(point_in_laser.point.y, point_in_laser.point.x)

            angle_index = int((angle + self.angle_comp) / self.lidar_data.angle_increment)
            angle_index = max(0, min(angle_index, len(self.lidar_data.ranges) - 1))

            if angle_index == self.special_idx:
                real_lidar_dis = self.special_dist
            else:
                real_lidar_dis = self.lidar_data.ranges[angle_index]

            if np.isinf(real_lidar_dis) or np.isnan(real_lidar_dis):
                rospy.logwarn_throttle(2, f"激光雷达在索引 {angle_index} 处距离无效，跳过。")
                return

            virtual_lidar_dis = np.hypot(point_in_laser.point.x, point_in_laser.point.y)

            #rospy.loginfo(f"激光雷达距离: {real_lidar_dis:.4f} m | 像素反算距离: {virtual_lidar_dis:.4f} m")

            if virtual_lidar_dis > 1e-6:
                #scale = real_lidar_dis / virtual_lidar_dis
                # =======================【缩放倍数校验】=======================
                # a. 先计算出当前的scale值
                current_scale = real_lidar_dis / virtual_lidar_dis
                
                # b. 如果不是第一次运行，则进行检查
                if self.last_scale is not None:
                    # c. 如果变化过大，则打印警告并沿用旧值
                    if abs(current_scale - self.last_scale) > 1.3:
                        rospy.logwarn_throttle(2, f"激光雷达在索引 {angle_index} 处距离无效，跳过。")
                        scale = self.last_scale
                    # d. 如果变化在接受范围内，则更新scale值
                    else:
                        scale = current_scale
                        self.last_scale = scale # 只有在变化不大时才更新
                # e. 如果是第一次运行，则直接使用当前值
                else:
                    scale = current_scale
                    self.last_scale = scale
                # ===============================================================
                
                
                real_lidar_point = PointStamped()
                real_lidar_point.header = point_in_laser.header
                real_lidar_point.point.x = point_in_laser.point.x * scale
                real_lidar_point.point.y = point_in_laser.point.y * scale
                real_lidar_point.point.z = point_in_laser.point.z * scale

                transform_laser_to_base = self.tf_buffer.lookup_transform(
                    self.base_frame, self.lidar_frame, rospy.Time(0), rospy.Duration(0.1)
                )
                final_point = tf2_geometry_msgs.do_transform_point(real_lidar_point, transform_laser_to_base)

                self.base_point_pub.publish(final_point)
                rospy.loginfo(f"距小车中心: {final_point.point.x:.4f} m | 缩放后坐标: x={real_lidar_point.point.x:.4f}, y={real_lidar_point.point.y:.4f}, z={real_lidar_point.point.z:.4f} | 缩放前坐标: x={point_in_laser.point.x:.4f}, y={point_in_laser.point.y:.4f}, z={point_in_laser.point.z:.4f}")

        except tf2_ros.TransformException as e:
            rospy.logwarn(f"坐标变换失败: {e}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        transformer = RefactoredPointTransformer()
        transformer.run()
    except rospy.ROSInterruptException:
        pass