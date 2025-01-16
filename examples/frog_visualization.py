import time
import pickle
from typing import List

import tqdm
import numpy as np

import sdsl
from sdsl.visualization.viz2d import visualize_2d, voxel_to_segments

MAP_PATH = "resources/maps/frog.poly"
RAW_TRAJ_PATH = "data/data2d/ros_frog/bag_files/traj_raw.pkl"

def read_poses():
    poses = []
    with open(RAW_TRAJ_PATH, "rb") as fp:
        records = pickle.load(fp)
    for record in records:
        if record["topic"] != "/pose":
            continue
        time_ns = record["time_ns"]
        position = record["pose"]["pose"]["position"]
        orientation = record["pose"]["pose"]["orientation"]
        poses.append((position["x"], position["y"]))
    return np.array(poses)

if __name__ == "__main__":
    env = sdsl.load_poly_file(MAP_PATH)

    poses = read_poses()
    segments = []
    for i in range(1, len(poses)):
        segments.append([poses[i-1][0], poses[i-1][1], poses[i][0], poses[i][1]])

    visualize_2d(env, blue_segments=np.array(segments))