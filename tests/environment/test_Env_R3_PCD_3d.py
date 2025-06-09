import numpy as np
from scipy.spatial import cKDTree

import sdsl


TEST_FILE = "resources/maps/3d/lab446a.ply"
TOLERANCE = 1e-7
ARR = sdsl.loaders.load_pcd_3d(TEST_FILE)

def test_Env_R3_PCD_init_3d():
    env = sdsl.Env_R3_PCD(ARR)
    assert np.allclose(
        np.sort(env.get_representation(), axis=0),
        np.sort(ARR, axis=0),
        atol=TOLERANCE
    )


def test_Env_R3_PCD_measure_distance_3d():
    env = sdsl.Env_R3_PCD(ARR)
    dirs = [(1,0,0), (0,1,0), (-1,0,0), (0,-1,0), (0,0,-1)]
    dists = [2.77911139, 0.73157191, 0.97499287, 2.43388128, 0.88810048]
    for dr, dist_ in zip(dirs, dists):
        ray = sdsl.R3xS2(0,0,0.9,*dr)
        dist = env.measure_distance(ray)
        assert abs(dist - dist_) < TOLERANCE
    
    # Cast a lot of random rays and check distances
    tree = cKDTree(ARR)
    num_inf = 0
    for _ in range(100):
        v = np.random.rand(3) * 2 - 1
        v /= np.linalg.norm(v)
        a, b, c = v[0], v[1], v[2]

        v = np.random.rand(3) * 2 - 1
        v *= 0.3
        v[2] += 0.9
        d, e, f = v[0], v[1], v[2]

        ray = sdsl.R3xS2(d,e,f,a,b,c)
        dist = env.measure_distance(ray)
        if dist > 100:
            num_inf += 1
            continue

        query_pt = np.array([d + a * dist, e + b * dist, f + c * dist])
        min_dist, _ = tree.query(query_pt)
        assert min_dist < 0.05

    assert num_inf > 0 # It is unlikely that you will miss the blindspots in this pcd

def test_Env_R3_PCD_hausdorff_distance_2d():
    env = sdsl.Env_R3_PCD(ARR)

    d1 = env.hausdorff_distance(sdsl.R3xS1(0, 0, 1, 0))
    d2 = env.hausdorff_distance(sdsl.R3xS1(2, 1, 0.5, 0))
    d3 = env.hausdorff_distance(sdsl.R3xS1(-1.8, -0.9, 0.9, 0))

    assert np.allclose([d1, d2, d3], [0.695282344436283, 0.1209748463061244, 0.7940803411478589], atol=TOLERANCE)


def test_Env_R3_PCD_intersects_3d():
    env = sdsl.Env_R3_PCD(ARR)

    v1 = sdsl.R3xS1_Voxel(
        sdsl.R3xS1(0.7, 1.0, 1.5, 0),
        sdsl.R3xS1(0.8, 1.1, 1.6, 0)
    )
    v2 = sdsl.R3xS1_Voxel(
        sdsl.R3xS1(-0.1, -0.1, -0.1, 0),
        sdsl.R3xS1(0.1, 0.1, 0.1, 0)
    )
    assert not env.intersects(v1)
    assert env.intersects(v2)



if __name__ == "__main__":
    env = sdsl.Env_R3_PCD(ARR)
    
    lines = []
    dirs = [(1,0,0), (0,1,0), (-1,0,0), (0,-1,0), (0,0,-1)]

    for dr in dirs:
        ray = sdsl.R3xS2(0,0,0.9,*dr)
        dist = env.measure_distance(ray)
        if (dist > 100):
            dist = 100

        s = np.array([ray.x(), ray.y(), ray.z()])
        v = np.array([ray.v1(), ray.v2(), ray.v3()])
        t = s + dist * v
        print(s, t)

        line = sdsl.visualization.visualize_line_3d(s, t)
        lines.append(line)

    v1 = sdsl.R3xS1_Voxel(
        sdsl.R3xS1(0.7, 1.0, 1.5, 0),
        sdsl.R3xS1(0.8, 1.1, 1.6, 0)
    )
    lines += sdsl.visualization.visualize_voxel_3d(v1)

    sdsl.visualization.visualize_pcd_3d(env, lines)

