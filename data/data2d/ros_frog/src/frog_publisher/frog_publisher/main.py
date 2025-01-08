import os

import h5py
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped

PUBLISH_DELAY = 0.01
FROG_H5 = os.environ["FROG_RAW_DATA"]
FROG_ODOM1 = os.environ["FROG_ODOM1"]
FROG_ODOM2 = os.environ["FROG_ODOM2"]

class FrogPublisher(Node):
    TYPE_ODOM = 0
    TYPE_SCAN = 1

    def __init__(self):
        super().__init__('frog_publisher')
        # First setup the data
        self.load_frog_data()
        self.preprocess_events()

        # Then setup the ROS publishers
        self.scan_publisher_ = self.create_publisher(LaserScan, 'scan', 5)
        self.odom_publisher_ = self.create_publisher(Odometry, 'odom', 5)
        self.timer = self.create_timer(PUBLISH_DELAY, self.timer_callback)
        self.i = 0
        self.is_done = False

    def timer_callback(self):
        if self.i >= len(self.events):
            if not self.is_done:
                self.get_logger().info('(!)Finished publishing all scans and odometries(!)')
                self.is_done = True
            return
        
        _, type, i = self.events[self.i]
        self.i += 1
        if type == self.TYPE_SCAN:
            self.publish_scans(i)
        elif type == self.TYPE_ODOM:
            self.publish_odom(i)
        
        
    def publish_odom(self, i):
        odom = self.odom_data[i]
        self.get_logger().info('TYPE %s' % (str(type(odom)) + " " + str(type(odom[0]))))
        msg = Odometry()
        msg.header.frame_id = 'odom'
        msg.header.stamp = self.get_clock().now().to_msg()
        # msg.child_frame_id = 'odom'
        msg.pose.pose.position.x = float(odom[0])
        msg.pose.pose.position.y = float(odom[1])
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = float(np.sin(odom[2] / 2))
        msg.pose.pose.orientation.w = float(np.cos(odom[2] / 2))
        self.odom_publisher_.publish(msg)
        self.get_logger().info('Publishing odometry #%d {%s}' % (self.i, "..."))

    def publish_scans(self, i):
        scans = self.scans[i]
        msg = LaserScan()
        msg.header.frame_id = 'laser_frame'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.angle_min = -np.pi * 0.5
        msg.angle_max = np.pi * 0.5
        msg.angle_increment = np.pi / scans.shape[0]
        msg.time_increment = 1 / scans.shape[0]
        # msg.scan_time = self.timestamps[self.i]
        # scan_time is float:
        # msg.scan_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9
        msg.scan_time = 1.0
        msg.range_min = 0.2
        msg.range_max = 10.0
        msg.ranges = list(map(float, scans))
        msg.intensities = []
        self.scan_publisher_.publish(msg)
        self.get_logger().info('Publishing scan #%d {%s}' % (self.i, "..."))


    def load_frog_data(self):
        self.get_logger().info('Loading frog data...')
        hf = h5py.File(FROG_H5, 'r')
        self.scans = hf["scans"]
        self.circle_idx = hf["circle_idx"]
        self.circle_num = hf["circle_num"]
        self.circles = hf["circles"]
        self.timestamps = hf["timestamps"]

        odom1 = np.load(FROG_ODOM1)
        odom2 = np.load(FROG_ODOM2)
        self.odom_ts = np.concatenate((odom1["ts"], odom2["ts"]))
        self.odom_data = np.concatenate((odom1["data"], odom2["data"]))

    def preprocess_events(self):
        self.events = []
        for i in range(self.scans.shape[0]):
            self.events.append((self.timestamps[i], self.TYPE_SCAN, i))
        for i in range(self.odom_ts.shape[0]):
            self.events.append((self.odom_ts[i], self.TYPE_ODOM, i))
        self.events.sort(key=lambda x: x[0])



def main(args=None):
    rclpy.init(args=args)
    frog_publisher = FrogPublisher()
    rclpy.spin(frog_publisher)

    frog_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()