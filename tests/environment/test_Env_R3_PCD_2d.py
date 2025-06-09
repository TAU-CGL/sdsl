import numpy as np

import sdsl


TEST_FILE = "resources/maps/2d/lab_lidar.poly"
TOLERANCE = 1e-7


def test_Env_R3_PCD_init_2d():
    arr = sdsl.loaders.load_pcd_2d(TEST_FILE)
    env = sdsl.Env_R3_PCD(arr)
    assert np.allclose(
        np.sort(env.get_representation(), axis=0), 
        np.sort(arr, axis=0), 
        atol=TOLERANCE
    )

def test_Env_R3_PCD_measure_distance_2d():
    arr = sdsl.loaders.load_pcd_2d(TEST_FILE)
    env = sdsl.Env_R3_PCD(arr)
    ray = sdsl.R3xS2(0, 0, 0, 1, 0, 0)
    dist = env.measure_distance(ray)
    assert abs(dist - 3.0389390000000005) < TOLERANCE

    # Cast a lot of random rays and check distances
    num_inf = 0
    for _ in range(1000):
        v = np.random.rand(2) * 2 - 1
        v /= np.linalg.norm(v)
        a, b = v[0], v[1]

        v = np.random.rand(2) * 2 - 1
        v *= 0.2
        c, d = v[0], v[1]

        ray = sdsl.R3xS2(c, d, 0, a, b, 0)
        dist = env.measure_distance(ray)
        if dist > 100:
            num_inf += 1
            continue

        query_pt = np.array([c + a * dist, d + b * dist])
        min_dist = None
        for i in range(arr.shape[0]):
            pt = arr[i, :2]
            tmp = np.linalg.norm(pt - query_pt)
            if min_dist is None or tmp < min_dist:
                min_dist = tmp
        assert min_dist < 0.05
    
    assert num_inf == 0

def test_Env_R3_PCD_hausdorff_distance_2d():
    arr = sdsl.loaders.load_pcd_2d(TEST_FILE)
    env = sdsl.Env_R3_PCD(arr)

    d1 = env.hausdorff_distance(sdsl.R3xS1(0, 0, 0, 0))
    d2 = env.hausdorff_distance(sdsl.R3xS1(2, 1, 0, 0))
    d3 = env.hausdorff_distance(sdsl.R3xS1(-1.8, -0.9, 0, 0))

    assert np.allclose([d1, d2, d3], [0.58447737, 0.311576485, 0.161913577], atol=TOLERANCE)

def test_Env_R3_PCD_intersects_2d():
    arr = sdsl.loaders.load_pcd_2d(TEST_FILE)
    env = sdsl.Env_R3_PCD(arr)

    v1 = sdsl.R3xS1_Voxel(
        sdsl.R3xS1(-0.1, -0.1, 0, 0),
        sdsl.R3xS1(0.1, 0.1, 0, 0)
    )
    v2 = sdsl.R3xS1_Voxel(
        sdsl.R3xS1(0.7, 1.1, 0, 0),
        sdsl.R3xS1(-1.4, -1.1, 0, 0)
    )
    v3 = sdsl.R3xS1_Voxel(
        sdsl.R3xS1(2.8, 0.7, 0, 0),
        sdsl.R3xS1(2.9, 0.8, 0, 0)
    )
    v4 = sdsl.R3xS1_Voxel(
        sdsl.R3xS1(-2.7, -1.9, -1, 2),
        sdsl.R3xS1(3.6, 2.9, 1, 2)
    )

    assert not env.intersects(v1)
    assert env.intersects(v2)
    assert not env.intersects(v3)
    assert env.intersects(v4)

def test_Env_R3_PCD_is_inside_2d():
    arr = sdsl.loaders.load_pcd_2d(TEST_FILE)
    env = sdsl.Env_R3_PCD(arr)

    q1 = sdsl.R3xS1(0, 0, 0, 0)
    q2 = sdsl.R3xS1(3, 1, 0, 0)
    q3 = sdsl.R3xS1(-2, 2, 0, 0)
    q4 = sdsl.R3xS1(-0.55, 2.21, 0, 0)
    q5 = sdsl.R3xS1(0.19, -1.24, 0, 0)
    q6 = sdsl.R3xS1(2.73, -0.61, 0, 0)
    q7 = sdsl.R3xS1(2, -1, 0, 0)
    q8 = sdsl.R3xS1(0.55, -1.14, 0, 0)
    q9 = sdsl.R3xS1(0, 2, 0, 0)

    assert env.is_inside(q1)
    assert not env.is_inside(q2)
    assert env.is_inside(q3)
    assert env.is_inside(q4)
    assert env.is_inside(q5)
    assert env.is_inside(q6)
    assert not env.is_inside(q7)
    assert not env.is_inside(q8)
    assert not env.is_inside(q9)

def fdg(d: float, g: sdsl.R3xS2, q: sdsl.R3xS1) -> sdsl.R3xS1:
    q = np.array([q.x(), q.y(), q.z(), q.r()])
    gp = np.array([g.x(), g.y(), g.z()])
    gv = np.array([g.v1(), g.v2(), g.v3()])
    return sdsl.R3xS1(
        q[0] + (gp[0] + d * gv[0]) * np.cos(q[3]) + (-gp[1] - d * gv[1]) * np.sin(q[3]),
        q[1] + (gp[1] + d * gv[1]) * np.cos(q[3]) + (gp[0] + d * gv[0]) * np.sin(q[3]),
        q[2] + (gp[2] + d * gv[2]),
        q[3]
    )

def test_Env_R3_PCD_forward_2d():
    arr = sdsl.loaders.load_pcd_2d(TEST_FILE)
    env = sdsl.Env_R3_PCD(arr)

    g = sdsl.R3xS2(0.1, 0, 0, 1, 0, 0) # Sensor is mounted on robot, pointed initialy to the right

    d = 1.5

    xmin = -0.1; xmax = 0.3
    ymin = -0.2; ymax = 0.2
    rmin = 0.4; rmax = 0.8

    voxel = env.forward(d, g, sdsl.R3xS1_Voxel(
        sdsl.R3xS1(xmin, ymin, 0, rmin),
        sdsl.R3xS1(xmax, ymax, 0, rmax)
    ))

    for _ in range(10000):
        qx = np.random.uniform(xmin, xmax)
        qy = np.random.uniform(ymin, ymax)
        qr = np.random.uniform(rmin, rmax)
        q = sdsl.R3xS1(qx, qy, 0, qr)
        q_ = fdg(d, g, q)
        assert voxel.contains(q_)



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    while True:
        arr = sdsl.loaders.load_pcd_2d(TEST_FILE)
        env = sdsl.Env_R3_PCD(arr)
        fig, ax = sdsl.visualization.visualize_pcd_2d(env)

        v = np.random.rand(2) * 2 - 1
        v /= np.linalg.norm(v)
        a, b = v[0], v[1]

        v = np.random.rand(2) * 2 - 1
        v *= 0.2
        c, d = v[0], v[1]

        ray = sdsl.R3xS2(c, d, 0, a, b, 0)
        dist = env.measure_distance(ray)
        if dist > 100:
            dist = 100

        ax.plot([c, c + dist * a], [d, d + dist * b], 'r-')

        plt.show()