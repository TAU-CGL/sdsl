from typing import List
import numpy as np
import sdsl


TEST_FILE = "resources/maps/lab_lidar.poly"
TOLERANCE = 1e-7


def sample_q0(env: sdsl.Env_R3_PCD):
    """
    Sample random valid configuration that is inside the environment
    """
    MAX_TRY = 1000
    cnt = 0
    bb = env.bounding_box()
    while True:
        q0 = bb.sample()
        if env.is_inside(q0):
            return q0
        cnt += 1
        if cnt > MAX_TRY:
            raise RuntimeError("Could not sample valid q0 configuration inside environment.")

def test_localize_R3_pcd_2d():
    arr = sdsl.loaders.load_pcd_2d(TEST_FILE)
    env = sdsl.Env_R3_PCD(arr)

    L = 0.05
    odometry = []
    for i in range(32):
        odometry.append(sdsl.R3xS2(
            L * np.cos(i * np.pi / 16), L * np.sin(i * np.pi / 16), 0,
            np.cos(i * np.pi / 16), np.sin(i * np.pi / 16), 0
        ))

    for _ in range(100):
        q0 = sample_q0(env)
        measurements = []
        for g in odometry:
            measurements.append(env.measure_distance(q0.act(g)))
        
        # Skip infeasible locations
        if np.max(measurements) > 100:
            continue
        
        localization = sdsl.localize_R3_pcd(env, odometry, measurements, 0.05, 8)
        flag = False
        for v in localization:
            if v.contains(q0):
                flag = True
                break
        assert flag
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    arr = sdsl.loaders.load_pcd_2d(TEST_FILE)
    env = sdsl.Env_R3_PCD(arr)

    L = 0.05
    odometry = []
    for i in range(32):
        odometry.append(sdsl.R3xS2(
            L * np.cos(i * np.pi / 16), L * np.sin(i * np.pi / 16), 0,
            np.cos(i * np.pi / 16), np.sin(i * np.pi / 16), 0
        ))

    while True:
        fig, ax = sdsl.visualization.visualize_pcd_2d(env)

        q0 = sample_q0(env)
        measurements = []
        for g in odometry:
            g_ = q0.act(g)
            d = env.measure_distance(g_)
            measurements.append(d)
            color = "r-"
            if d > 100:
                d = 100
                color = "b-"
            ax.plot([g_.x(), g_.x() + d * g_.v1()], [g_.y(), g_.y() + d * g_.v2()], color)


        res = sdsl.localize_R3_pcd(env, odometry, measurements, 0.05, 8)
        print(res)

        
        plt.show()




