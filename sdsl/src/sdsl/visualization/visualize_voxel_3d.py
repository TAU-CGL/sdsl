import numpy as np
import open3d as o3d


def visualize_line_3d(p1: np.array, p2: np.array, color: np.array = np.array([1,0,0])) -> o3d.geometry.LineSet:
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector([p1, p2])
    line.lines = o3d.utility.Vector2iVector([[0,1]])
    line.colors = o3d.utility.Vector3dVector([color])
    return line