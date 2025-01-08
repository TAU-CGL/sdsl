Some of the data processing of the FROG dataset is done with the use of some ros packages.
More specifically, the navigation2 stack.
https://roboticsbackend.com/ros2-nav2-generate-a-map-with-slam_toolbox/

This was run on Apple Silicon (M1) computer, with RoboStack, and ROS2 (Humble), with mamba.

Don't forget to run `colcon build` from this root folder, activate ROS environment.
Also install dependencies: `mamba install h5py numpy`
In another terminals, run the navigation2 stack and slam toolbox.
Then, to run the publisher, use `./start.zsh`.