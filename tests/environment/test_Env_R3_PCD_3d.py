import numpy as np

import sdsl


TEST_FILE = "resources/maps/3d/lab446a.ply"
TOLERANCE = 1e-7

def test_Env_R3_PCD_init_3d():
    arr = sdsl.loaders.load_pcd_3d(TEST_FILE)
    env = sdsl.Env_R3_PCD(arr)
    assert np.allclose(
        np.sort(env.get_representation(), axis=0),
        np.sort(arr, axis=0),
        atol=TOLERANCE
    )

if __name__ == "__main__":
    arr = sdsl.loaders.load_pcd_3d(TEST_FILE)
    env = sdsl.Env_R3_PCD(arr)
    sdsl.visualization.visualize_pcd_3d(env)