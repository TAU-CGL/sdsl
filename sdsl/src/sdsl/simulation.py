import numpy as np

from ._sdsl import Environment

def corrupt_measurements(dists: np.ndarray, kk_prime_ratio: float, noise_eps: float):
    noisy = dists.copy()
    n_corrupt = round((1-kk_prime_ratio) * len(dists)) - 1
    if n_corrupt > 0:
        idx = np.random.choice(len(dists), size=n_corrupt, replace=False)
        noisy[idx] *= np.random.uniform(0.1, 3.0, size=n_corrupt)
    noisy += np.random.normal(0.0, 0.5 * noise_eps, size=len(noisy))
    return noisy

