import time
import pickle

import sdsl
from sdsl.simulation import *

MAP_PATH = "resources/maps/frog.poly"
ARROW_LEN = 0.3
K = 16; K_ = 14
EPS = 0.02
RECURSION_DEPTH = 8
SENSOR_OFFSET = 0.01

PICKLE_PATH = "resources/obstacles/scenarios.pkl"

if __name__ == "__main__":
    sdsl.seed(100)
    env = sdsl.load_poly_file(MAP_PATH)

    with open(PICKLE_PATH, "rb") as fp:
        scenarios = pickle.load(fp)
    
    for _ in range(1000):
        idx = np.random.randint(len(scenarios))
        q0_, dynamic_obstacles, odometry_ = scenarios[idx]

        q0 = R2xS1(q0_[0], q0_[1], q0_[2])
        odometry = [R2xS1(g[0], g[1], g[2]) for g in odometry_]
        measurements = get_measurements(env, odometry, q0, dynamic_obstacles)

        start = time.time()
        print("Starting localization...")
        localization = sdsl.localize_R2_dynamic_naive(env, odometry, measurements, EPS, RECURSION_DEPTH, K_)
        end = time.time()
        print(f"End localization. Took {end - start}[sec]")

        # visualize_simulation(env, q0, odometry, measurements, dynamic_obstacles)

        segments = []
        for voxel in localization:
            segments += voxel_to_segments(voxel)
            r = voxel.bottom_left().r()
            x = voxel.bottom_left().x()
            y = voxel.bottom_left().y()
            segments += [[x,y,x+ARROW_LEN * np.cos(r), y+ARROW_LEN * np.sin(r)]]

        blue_segments = [[q0.x(), q0.y(), q0.x() + ARROW_LEN * np.cos(q0.r()), q0.y() + ARROW_LEN * np.sin(q0.r())]]
        for g, d in zip(odometry, measurements):
            blue_segments += [[q0.x(), q0.y(), q0.x() + d * np.cos(q0.r() + g.r()), q0.y() + d * np.sin(q0.r() + g.r())]]

        visualize_2d(env, dynamic_obstacles=dynamic_obstacles, red_segments=np.array(segments), blue_segments=np.array(blue_segments),points=np.array([[q0.x(), q0.y()]]))
