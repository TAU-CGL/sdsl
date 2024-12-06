import numpy as np

import sdsl


if __name__ == "__main__":
    a = np.array([[1, 2, 3, 4]], dtype=np.uint64)
    env = sdsl.Env_R2(a)
    print(env.get_representation())