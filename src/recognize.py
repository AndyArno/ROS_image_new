#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped

class RefactoredYOLODetector:
    def __init__(self):
        """
        YOLO图像检测器
        从ROS参数服务器加载配置
        """
        rospy.init_node('recognize_node', anonymous=False)
        
        # 加载参数
        self.load_params()
        
        self.bridge = CvBridge()
        self.yolo_model = YOLO(self.model_path)
        
        # 订阅与发布
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)
        self.optical_point_pub = rospy.Publisher(self.optical_point_topic, PointStamped, queue_size=10)
        self.pixel_point_pub = rospy.Publisher(self.pixel_point_topic, PointStamped, queue_size=10)
        
        rospy.loginfo("YOLO识别节点启动成功。")

    def load_params(self):
        """从ROS参数服务器加载所有配置"""
        try:
            # 基础参数
            self.image_topic = rospy.get_param('~image_topic')
            self.model_path = rospy.get_param('~yolo_model_path')
            self.optical_point_topic = rospy.get_param('~output_optical_point_topic')
            self.pixel_point_topic = rospy.get_param('~output_pixel_point_topic')
            self.visualize = rospy.get_param('~visualize', True)
            
            # 相机内参
            cam_info = rospy.get_param('~camera_info')
            self.camera_matrix = np.array(cam_info['camera_matrix']['data']).reshape(3, 3)
            self.dist_coeffs = np.array(cam_info['distortion_coefficients']['data'])
            
            self.optical_frame_id = cam_info.get('optical_frame_id', 'camera_depth_optical_frame')

            rospy.loginfo("相机参数加载成功。")
        except KeyError as e:
            rospy.logerr(f"参数加载失败，请检查launch文件和yaml文件: {e}")
            rospy.signal_shutdown("缺少关键参数")

    def image_callback(self, msg):
        """图像处理回调函数"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # 1. 图像去畸变 (关键改进)
        undistorted_image = cv2.undistort(cv_image, self.camera_matrix, self.dist_coeffs)
        
        # 2. YOLO推理
        results = self.yolo_model(undistorted_image, verbose=False)
        
        # 3. 处理并发布结果
        if results and results[0].boxes:
            box = results[0].boxes.xyxy[0].tolist()
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            
            # 发布去畸变后的像素坐标
            self.publish_point(self.pixel_point_pub, msg.header, "pixel_frame", [x_center, y_center, 0])
            
            # 像素坐标到归一化相机坐标
            pixel_coords = np.array([x_center, y_center, 1.0])
            k_inv = np.linalg.inv(self.camera_matrix)
            optical_coords = k_inv.dot(pixel_coords)
            
            self.publish_point(self.optical_point_pub, msg.header, self.optical_frame_id, optical_coords)

        # 4. 可视化
        if self.visualize:
            annotated_frame = results[0].plot()
            cv2.imshow("YOLO Detection (Undistorted)", annotated_frame)
            cv2.waitKey(1)

    def publish_point(self, publisher, header, frame_id, coords):
        """通用坐标点发布函数"""
        point_msg = PointStamped()
        point_msg.header = header # 保持与源图像相同的时间戳
        point_msg.header.frame_id = frame_id
        point_msg.point.x = coords[0]
        point_msg.point.y = coords[1]
        point_msg.point.z = coords[2]
        publisher.publish(point_msg)

    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("节点关闭。")
        finally:
            if self.visualize:
                cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        detector = RefactoredYOLODetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass