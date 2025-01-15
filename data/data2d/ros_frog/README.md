Some of the data processing of the FROG dataset is done with the use of some ros packages.
More specifically, the navigation2 stack.
https://roboticsbackend.com/ros2-nav2-generate-a-map-with-slam_toolbox/

This was run on Apple Silicon (M1) computer, with RoboStack, and ROS2 (Humble), with mamba.

Don't forget to run `colcon build` from this root folder, activate ROS environment.
Also install dependencies: `mamba install h5py numpy`.
Install (into the mamba environment) the `nml_bad` repository: https://github.com/ricmua/nml_bag

## Map Generation

On all terminals, `cd` into the `ros_frog` directory, and run `mamba activate ros_env`.
First, run this as normal.
Then, to add subsequent maps, you need to restart terminal 3 with different point clouds.
Use the `slam_toolbox` Rviz plugin to reset the robot's location back to dock.

Terminal 1:
    ```
    rviz2
    ```

Terminal 2:
    ```
    ./map_start.zsh
    ```

Terminal 3: First make sure to edit the `env.zsh` script to point at the correct dataset.
    ```
    source ./env.zsh
    ./start.zsh
    ```

You can run `maps/pgm_to_poly.py` to post process the map into an sdsl comptaible *.poly file.

## Localization

After mapping, you can apply localization and download into rosbags:

Terminal 1:
    ```
    rviz2
    ```

Terminal 2:
    ```
    ./localize_start.zsh
    ```

Terminal 3: First make sure to edit the `env.zsh` script to point at the correct dataset.
    ```
    source ./env.zsh
    ./start.zsh
    ```

Terminal 4:
    ```
    ros2 bag record -o bag_files/traj -s mcap /scan_idx /pose
    ```

Then run `bag_files/process_traj.py` to create the h5 file of the localized trajectory with correlations to original point clouds.