"""
Manual visualization test for localize function with unit sphere surface predicate.

Run this directly (not via pytest) to visualize the voxels that intersect
the unit sphere surface (S^2) after 8 levels of recursion.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

import sdsl


def box_edges_3d(bl, tr):
    """Generate wireframe edges for a 3D box."""
    x0, y0, z0 = bl[0], bl[1], bl[2]
    x1, y1, z1 = tr[0], tr[1], tr[2]
    
    corners = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)
    ]
    
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]
    
    return [(corners[i], corners[j]) for i, j in edges]


def plot_voxels_3d(voxels, ax, color='tab:blue', alpha=1.0, linewidth=0.5):
    """Plot a collection of 3D voxels as wireframes."""
    lines = []
    for v in voxels:
        bl = (v.bottom_left[0], v.bottom_left[1], v.bottom_left[2])
        tr = (v.top_right[0], v.top_right[1], v.top_right[2])
        lines.extend(box_edges_3d(bl, tr))
    
    lc = Line3DCollection(lines, colors=color, linewidths=linewidth, alpha=alpha)
    ax.add_collection3d(lc)


def plot_unit_sphere(ax, resolution=50):
    """Plot the unit sphere as a wireframe."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, color='tab:red', alpha=0.2, edgecolor='none')
    ax.plot_wireframe(x, y, z, color='tab:red', alpha=0.3, linewidth=0.3, rcount=10, ccount=10)


def main():
    print("Running unit sphere surface (S^2) localization with recursion depth 8...")
    print("This may take a moment due to OpenMP parallelization and visualization.\n")
    
    # Create bounding box that contains the unit sphere
    bl = sdsl.Config_3d()
    tr = sdsl.Config_3d()
    
    # Bounding box from -1.5 to 1.5 in each dimension
    for i in range(3):
        bl[i] = -1.5
        tr[i] = 1.5
    
    bounding_box = sdsl.Voxel_3d(bl, tr)
    
    # Run localization with recursion depth 8
    recursion_depth = 5
    print(f"Bounding box: [{bl[0]}, {tr[0]}]^3")
    print(f"Recursion depth: {recursion_depth}")
    print(f"Maximum voxels per level: {8 ** recursion_depth:,}\n")
    
    voxels = sdsl.localize_unit_sphere(bounding_box, recursion_depth)
    
    print(f"\nLocalization complete!")
    print(f"Number of voxels intersecting unit sphere surface: {len(voxels):,}")
    
    # Calculate average voxel size
    if len(voxels) > 0:
        sample_voxel = voxels[0]
        voxel_size = sample_voxel.top_right[0] - sample_voxel.bottom_left[0]
        print(f"Voxel size at depth {recursion_depth}: {voxel_size:.6f}")
    
    # Visualize
    print("\nGenerating visualization...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the unit sphere
    plot_unit_sphere(ax)
    
    # Plot the voxels
    plot_voxels_3d(voxels, ax, color='tab:blue', alpha=0.6, linewidth=0.4)
    
    # Setup axes
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'Unit Sphere Surface (S²) Localization (Depth {recursion_depth}, {len(voxels):,} voxels)', 
                 fontsize=14, pad=20)
    
    # Set equal aspect ratio
    max_range = 1.6
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='tab:red', alpha=0.3, label='Unit Sphere Surface (S²)'),
        Patch(facecolor='tab:blue', alpha=0.6, label='Intersecting Voxels')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    print("Displaying plot...")
    plt.show()


if __name__ == "__main__":
    main()
