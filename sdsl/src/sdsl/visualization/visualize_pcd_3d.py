from typing import List

import numpy as np
import open3d as o3d

from sdsl import Env_R3_PCD


def visualize_pcd_3d(env: Env_R3_PCD, extra_geometries = None):
    points = env.get_representation()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    geom = [pcd]
    if extra_geometries is not None:
        geom += extra_geometries
    o3d.visualization.draw_geometries(geom)