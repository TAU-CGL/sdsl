from typing import List

import numpy as np
import open3d as o3d

from sdsl import R3xS1_Voxel


def visualize_line_3d(p1: np.array, p2: np.array, color: np.array = np.array([1,0,0])) -> o3d.geometry.LineSet:
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector([p1, p2])
    line.lines = o3d.utility.Vector2iVector([[0,1]])
    line.colors = o3d.utility.Vector3dVector([color])
    return line

def visualize_voxel_3d(v: R3xS1_Voxel, color: np.array = np.array([0,1,0])) -> List[o3d.geometry.LineSet]:
    bl = v.bottom_left(); tr = v.top_right()
    bl = [bl.x(), bl.y(), bl.z()]
    tr = [tr.x(), tr.y(), tr.z()]
    points = [
        [bl[0], bl[1], bl[2]], #p1
        [tr[0], bl[1], bl[2]], #p2
        [tr[0], tr[1], bl[2]], #p3
        [bl[0], tr[1], bl[2]], #p4
        [bl[0], bl[1], tr[2]], #p5
        [tr[0], bl[1], tr[2]], #p6
        [tr[0], tr[1], tr[2]], #p7
        [bl[0], tr[1], tr[2]], #p8
    ]
    indices = [
        (1,2),(2,3),(3,4),(4,1),
        (5,6),(6,7),(7,8),(8,5),
        (1,5),(2,6),(3,7),(4,8),
    ]
    return [visualize_line_3d(points[i1-1], points[i2-1], color) for i1,i2 in indices]