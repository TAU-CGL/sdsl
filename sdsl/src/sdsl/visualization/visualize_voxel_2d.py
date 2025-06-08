import matplotlib.pyplot as plt

import sdsl

def visualize_voxel_2d(ax, v: sdsl.R3xS1_Voxel, color: str = "#00FF00"):
    bl = v.bottom_left(); tr = v.top_right()
    Xs = [bl.x(), bl.x(), tr.x(), tr.x(), bl.x()]
    Ys = [bl.y(), tr.y(), tr.y(), bl.y(), bl.y()]
    ax.plot(Xs, Ys, color=color)