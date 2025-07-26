#!/usr/bin/env python3
import math

import cv2
import rclpy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image

from lane_follower_pkg.model.unet import UNet
from lane_follower_pkg.postprocess import process_prediction
from lane_follower_pkg.transforms import apply_filters


class LaneFollowerNode(Node):
    def __init__(self):
        super().__init__("lane_follower")
        # params
        self.declare_parameter("camera_topic", "/camera/color/image_raw")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("model_path", "lane_unet.pth")
        self.declare_parameter("forward_speed", 0.2)
        self.declare_parameter("steer_gain", 0.01)
        self.declare_parameter("max_steer", 0.5)

        cam_topic = self.get_parameter("camera_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        cmd_topic = self.get_parameter("cmd_topic").value
        model_path = self.get_parameter("model_path").value
        self.v = self.get_parameter("forward_speed").value
        self.k = self.get_parameter("steer_gain").value
        self.max_w = self.get_parameter("max_steer").value

        # device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.get_logger().info("Using CUDA")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.get_logger().info("Using MPS")
        else:
            self.device = torch.device("cpu")
            self.get_logger().info("Using CPU")

        # load model
        self.model = UNet().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt)
        self.model.eval()

        # ROS interfaces
        self.bridge = CvBridge()
        self.sub_img = self.create_subscription(
            Image, cam_topic, self.image_callback, 1
        )
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 1
        )
        self.pub_twist = self.create_publisher(Twist, cmd_topic, 1)
        self.current_yaw = 0.0

    def odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        siny = 2 * (q.w * q.z + q.x * q.y)
        cosy = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny, cosy)

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        H, W = frame.shape[:2]

        # preprocess & inference
        small = cv2.resize(frame, (256, 256))
        filt = apply_filters(small)
        np_img = filt.astype("float32") / 255.0
        tensor = (
            torch.from_numpy(np_img.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            clean, lines, angle_deg = process_prediction(self.model(tensor))

        # compute steering
        steer = -angle_deg * (math.pi / 180.0) * self.k
        steer = max(-self.max_w, min(self.max_w, steer))

        twist = Twist()
        twist.linear.x = float(self.v)
        twist.angular.z = float(steer)
        self.pub_twist.publish(twist)

    def destroy_node(self):
        super().destroy_node()
        self.get_logger().info("Shutting down")


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
