import json
import argparse

import sdsl
from sdsl.simulation import *

parser = argparse.ArgumentParser(description="Exp: k-k'-gap in uniform sampled obstacles")
parser.add_argument("--map", type=str, default="resources/maps/square_room.poly")
parser.add_argument("--k", type=int, default=16)
parser.add_argument("--sensor-offset", type=float, default=0.01)
parser.add_argument("--obst-num", type=int, default=10)
parser.add_argument("--obst-radius", type=float, default=0.1)
parser.add_argument("--num-experiments", type=int, default=100)
parser.add_argument("--seed", type=int, default=100)
args = parser.parse_args()

if __name__ == "__main__":
    sdsl.seed(args.seed)
    env = sdsl.load_poly_file(args.map)

    results = {"k_": []}
    for _ in range(args.num_experiments):
        q0, odometry, dynamic_obstacles = get_valid_scenario(env, args.k, args.sensor_offset, args.obst_num, args.obst_radius)
        if q0 is None:
            results["k_"].append(-1)
            continue
        measurements = get_measurements(env, odometry, q0, dynamic_obstacles)
        k_ = 0
        for g in odometry:
            k_ += 1 if not is_outlier_measurement(env, g * q0, dynamic_obstacles) else 0
        results["k_"].append(k_)

    print(json.dumps(results))