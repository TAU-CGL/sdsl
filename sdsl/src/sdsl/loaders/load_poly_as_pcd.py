import numpy as np

def load_poly_as_pcd(path, spacing):
    pts = np.loadtxt(path)
    pts_closed = np.vstack([pts, pts[0]])

    samples = []
    for i in range(len(pts)):
        p0 = pts_closed[i]
        p1 = pts_closed[i+1]
        edge_len = np.linalg.norm(p1 - p0)
        n = max(2, int(np.round(edge_len / spacing)) + 1)
        ts = np.linspace(0.0, 1.0, n, endpoint=False)
        for t in ts:
            samples.append(p0 + t * (p1 - p0))
    
    return np.array(samples, dtype=np.float64)