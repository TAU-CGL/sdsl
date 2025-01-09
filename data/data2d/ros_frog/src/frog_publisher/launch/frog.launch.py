from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Node(
        #     package="tf2_ros",
        #     executable="static_transform_publisher",
        #     arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
        # ),
        # Node(
        #     package="tf2_ros",
        #     executable="static_transform_publisher",
        #     arguments=["0", "0", "0", "0", "0", "0", "odom", "base_link"],
        # ),
        # Node(
        #     package="tf2_ros",
        #     executable="static_transform_publisher",
        #     arguments=["0", "0", "0", "0", "0", "0", "odom", "base_footprint"],
        # ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["0", "0", "0", "0", "0", "0", "base_footprint", "laser_frame"],
        ),
        Node(
            package='frog_publisher',
            executable='publisher',
        )
    ])