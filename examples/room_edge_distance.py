from PIL import Image
import numpy as np
from scipy.ndimage import distance_transform_edt
import yaml

def load_map_config(yaml_path):
    """Load map configuration from YAML"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def point_to_room_edge_distance(pixels, current_point, target_room_rgb, resolution=0.05):
    """
    Calculate distance from current point to nearest edge of a room.

    Args:
        pixels: numpy array of image pixels (H, W, 3)
        current_point: (x, y) in pixel coordinates
        target_room_rgb: [R, G, B] list
        resolution: meters per pixel

    Returns:
        (distance_pixels, distance_meters)
    """
    # Create binary mask: True where room pixels match exactly
    mask = np.all(pixels == target_room_rgb, axis=2)

    # Distance transform: distance from each pixel to nearest room boundary
    distance_map = distance_transform_edt(~mask)

    x, y = int(current_point[0]), int(current_point[1])

    # Clamp to image bounds
    h, w = distance_map.shape
    if not (0 <= x < w and 0 <= y < h):
        return None, None

    distance_pixels = distance_map[y, x]
    distance_meters = distance_pixels * resolution

    return distance_pixels, distance_meters

if __name__ == "__main__":
    # Load map config
    config = load_map_config('resources/maps/2d/slam/simple_symmetry/symmetry_2.yaml')

    # Load image
    img = Image.open('resources/maps/2d/slam/simple_symmetry/symmetry_2.png')
    pixels = np.array(img)

    resolution = config['resolution']

    # Build room color dict
    room_colors = {room['name']: room['rgb'] for room in config['rooms']}

    # Example: point in room_left, distance to hall_left edge
    current_point = (100, 150)
    target_room = 'hall_left'

    dist_px, dist_m = point_to_room_edge_distance(
        pixels, current_point, room_colors[target_room], resolution
    )

    if dist_m is not None:
        print(f"Current point: {current_point}")
        print(f"Target room: {target_room}")
        print(f"Distance to edge: {dist_px:.2f} pixels = {dist_m:.2f} meters")
    else:
        print("Point out of bounds")
