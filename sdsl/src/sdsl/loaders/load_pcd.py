import trimesh
import numpy as np


def load_pcd_2d(filename: str) -> np.array:
    """
    Load a 2D point cloud from a PCD file, in the *.poly format (lines are "x y\n").
    Note that the PCD environment assumes 3D points, so we automatically set the z-coordinate to zero.

    Args:
        filename (str): Path to the PCD file.

    Returns:
        np.array: A 2D numpy array with shape (N, 3), where N is the number of points.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Read the points
    points = []
    for line in lines:
        if line.strip():  # Skip empty lines
            x, y = map(float, line.split()[:2])  # Only take the first two dimensions
            points.append([x, y, 0])

    return np.array(points, dtype=np.double)


def load_pcd_3d(filename: str) -> np.array:
    """
    Load a 3D point cloud from any file supported by trimesh. 
    In this method we only consider the vertices of the mesh file.

    Args:
        filename (str): Path to the PCD file.

    Returns:
        np.array: A 2D numpy array with shape (N, 3), where N is the number of points.
    """
    ply = trimesh.load(filename)
    return np.array(ply.vertices)