import time

import numpy as np

import sdsl

TEST_FILE = "resources/maps/3d/lab446a.ply"
ARR = sdsl.loaders.load_pcd_3d(TEST_FILE)

def test_localize_R3_pcd_3d():
    env = sdsl.Env_R3_PCD(ARR)
    # bb = env.bounding_box()
    bb = sdsl.R3xS1_Voxel(
        sdsl.R3xS1(-0.6, -0.6, 0.8, 0),
        sdsl.R3xS1(0.6, 0.6, 1.1, 2 * 3.14159265)
    )

    # Mimic the crazyfly measurement scheme
    odometry = [
        sdsl.R3xS2(0.05, 0, 0, 1, 0, 0),
        sdsl.R3xS2(-0.05, 0, 0, -1, 0, 0),
        sdsl.R3xS2(0, 0.05, 0, 0, 1, 0),
        sdsl.R3xS2(0, -0.05, 0, 0, -1, 0),
        sdsl.R3xS2(0, 0, -0.01, 0, 0, -1),
    ]

    num_fails = 0
    num_experiments = 100
    for _ in range(num_experiments):
        q0 = bb.sample()
        measurements = []
        for g in odometry:
            measurements.append(env.measure_distance(q0.act(g)))
        
        # Skip locations that sample voids
        if np.max(measurements) > 100:
            num_fails += 1
            continue

        localization = sdsl.localize_R3_pcd(env, odometry, measurements, 0.015, 8)
        flag = False
        for v in localization:
            if v.contains(q0):
                flag = True
                break
        assert flag
    
    # At least 10% should hopefuly be feasible place samples
    assert num_fails < 0.1 * num_experiments 


if __name__ == "__main__":
    env = sdsl.Env_R3_PCD(ARR)
    # bb = env.bounding_box()
    bb = sdsl.R3xS1_Voxel(
        sdsl.R3xS1(-0.6, -0.6, 0.8, 0),
        sdsl.R3xS1(0.6, 0.6, 1.1, 2 * 3.14159265)
    )
    odometry = [
        sdsl.R3xS2(0.05, 0, 0, 1, 0, 0),
        sdsl.R3xS2(-0.05, 0, 0, -1, 0, 0),
        sdsl.R3xS2(0, 0.05, 0, 0, 1, 0),
        sdsl.R3xS2(0, -0.05, 0, 0, -1, 0),
        sdsl.R3xS2(0, 0, -0.01, 0, 0, -1),
    ]

    while True:
        q0 = bb.sample()
        measurements = []
        for g in odometry:
            measurements.append(env.measure_distance(q0.act(g)))
        # Skip locations that sample voids
        if np.max(measurements) > 100:
            continue
        
        t0 = time.time()
        res = sdsl.localize_R3_pcd(env, odometry, measurements, 0.015, 8)
        t1 = time.time()
        print(f"Took {t1-t0}[sec]")

        extra_geom = []
        for g, d in zip(odometry, measurements):
            g_ = q0.act(g)
            v = np.array([g_.v1(), g_.v2(), g_.v3()])
            p1 = np.array([g_.x(), g_.y(), g_.z()])
            p2 = p1 + d * v
            extra_geom.append(sdsl.visualization.visualize_line_3d(p1, p2))
        for v in res:
            extra_geom += sdsl.visualization.visualize_voxel_3d(v)
        
        sdsl.visualization.visualize_pcd_3d(env, extra_geom)

