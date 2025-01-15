#!/bin/zsh
ros2 launch slam_toolbox online_async_launch.py use_sim_time:=True slam_params_file:="config/localization.yaml"