#!/bin/zsh

# MAKE SURE TO ADD THIS ENVIRONMENT VARIABLE
# export FROG_RAW_DATA="..."
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source ./install/setup.zsh
# ros2 run frog_publisher publisher
ros2 launch frog_publisher frog.launch.py