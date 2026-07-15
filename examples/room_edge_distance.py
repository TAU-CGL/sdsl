from PIL import Image
import numpy as np
from scipy.ndimage import distance_transform_edt
import yaml
import networkx as nx
import matplotlib.pyplot as plt

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
        current_point: (x, y) in pixel coordinates where x is vertical (row), y is horizontal (col)
        target_room_rgb: [R, G, B] list
        resolution: meters per pixel

    Returns:
        (distance_pixels, distance_meters)
    """
    # Create binary mask: True where room pixels match exactly
    mask = np.all(pixels == target_room_rgb, axis=2)

    # Distance transform: distance from each pixel to nearest room boundary
    distance_map = distance_transform_edt(~mask)

    x_pixel, y_pixel = int(current_point[0]), int(current_point[1])

    # Clamp to image bounds (x is row, y is column)
    h, w = distance_map.shape
    if not (0 <= x_pixel < h and 0 <= y_pixel < w):
        return None, None

    distance_pixels = distance_map[x_pixel, y_pixel]
    distance_meters = distance_pixels * resolution

    return distance_pixels, distance_meters

def create_room_graph(config):
    """Create a graph from room configuration"""
    graph = nx.Graph()

    for room in config['rooms']:
        graph.add_node(room['name'], rgb=room['rgb'])

    for edge in config['graph_edges']:
        graph.add_edge(edge[0], edge[1])

    return graph

def print_graph_info(graph):
    """Print graph statistics"""
    print(f"Rooms: {list(graph.nodes())}")
    print(f"Connections: {list(graph.edges())}")
    print(f"Number of rooms: {graph.number_of_nodes()}")
    print(f"Number of connections: {graph.number_of_edges()}")

def find_path(graph, start_room, end_room):
    """Find path between two rooms"""
    try:
        path = nx.shortest_path(graph, start_room, end_room)
        return path
    except nx.NetworkXNoPath:
        return None

def world_to_pixel(world_point, origin, resolution, img_height):
    """
    Convert world coordinates to pixel coordinates.

    Args:
        world_point: (world_x, world_y) - x is horizontal (right), y is vertical (up)
        origin: (origin_x, origin_y) in world coordinates
        resolution: meters per pixel
        img_height: height of image in pixels

    Returns:
        (pixel_x, pixel_y) - x is vertical (row, down), y is horizontal (column, right)
    """
    pixel_y = (world_point[0] - origin[0]) / resolution  # horizontal: world_x -> pixel_y
    pixel_x = img_height - (world_point[1] - origin[1]) / resolution  # vertical inverted: world_y -> pixel_x
    return (pixel_x, pixel_y)

def pixel_to_world(pixel_point, origin, resolution, img_height):
    """
    Convert pixel coordinates to world coordinates.

    Args:
        pixel_point: (pixel_x, pixel_y) - x is vertical (row, down), y is horizontal (column, right)
        origin: (origin_x, origin_y) in world coordinates
        resolution: meters per pixel
        img_height: height of image in pixels

    Returns:
        (world_x, world_y) - x is horizontal (right), y is vertical (up)
    """
    pixel_x, pixel_y = pixel_point
    world_x = pixel_y * resolution + origin[0]  # horizontal: pixel_y -> world_x
    world_y = (img_height - pixel_x) * resolution + origin[1]  # vertical inverted: pixel_x -> world_y
    return (world_x, world_y)

def get_room_at_point(pixels, current_point, room_colors):
    """
    Determine which room a point is in based on pixel color.

    Args:
        pixels: numpy array (H, W, 3)
        current_point: (x, y) in pixel coordinates where x is row (vertical), y is column (horizontal)
        room_colors: dict mapping room names to RGB tuples
    """
    x_pixel, y_pixel = int(current_point[0]), int(current_point[1])

    h, w = pixels.shape[:2]
    if not (0 <= x_pixel < h and 0 <= y_pixel < w):
        return None

    pixel_color = pixels[x_pixel, y_pixel]

    for room_name, room_rgb in room_colors.items():
        if np.array_equal(pixel_color, room_rgb):
            return room_name

    return None

def distances_to_neighbors(pixels, current_point, current_room, graph, room_colors, resolution):
    """
    Calculate distance from current point to edge of all neighboring rooms.

    Args:
        current_point: (x, y) in pixel coordinates where x is row (vertical), y is column (horizontal)
    """
    neighbors = list(graph.neighbors(current_room))

    if not neighbors:
        print(f"  No neighbors for {current_room}")
        return

    print(f"  Distances from point {current_point} to neighbors of {current_room}:")
    for neighbor in sorted(neighbors):
        dist_px, dist_m = point_to_room_edge_distance(
            pixels, current_point, room_colors[neighbor], resolution
        )
        if dist_m is not None:
            print(f"    -> {neighbor}: {dist_px:.2f} px = {dist_m:.3f} m")
        else:
            print(f"    -> {neighbor}: out of bounds")

def visualize_graph(graph, config, output_path='room_graph.png'):
    """Visualize the graph with room colors"""
    plt.figure(figsize=(10, 8))

    pos = nx.spring_layout(graph, seed=42, k=2)

    room_colors_dict = {room['name']: room['rgb'] for room in config['rooms']}
    node_colors = [
        [c / 255.0 for c in room_colors_dict[node]] for node in graph.nodes()
    ]

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=3000)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(graph, pos, width=2, alpha=0.6)

    plt.title('Room Connectivity Graph')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Graph saved to {output_path}")

if __name__ == "__main__":
    # Load map config
    config = load_map_config('resources/maps/2d/slam/simple_symmetry/symmetry_2_c.yaml')

    # Load image and convert to RGB (drops alpha channel if present)
    img = Image.open('resources/maps/2d/slam/simple_symmetry/symmetry_2_c.png').convert('RGB')
    pixels = np.array(img)

    resolution = config['resolution']
    origin = config['origin']
    room_colors = {room['name']: room['rgb'] for room in config['rooms']}
    img_height = pixels.shape[0]

    # Create graph
    graph = create_room_graph(config)
    print("=== Graph Info ===")
    print_graph_info(graph)
    print()

    # Find path example
    start = 'room_left'
    end = 'room_right'
    path = find_path(graph, start, end)
    if path:
        print(f"Path from {start} to {end}: {' -> '.join(path)}")
    else:
        print(f"No path between {start} and {end}")
    print()

    # Distance to all neighbors example (world coordinates in meters)
    print("=== Distances to Neighbors ===")
    current_point_world = (1, 1.12)  # world coordinates: (x_world, y_world) in meters
    current_point_pixel = world_to_pixel(current_point_world, origin[:2], resolution, img_height)

    # Debug
    print(f"Image height: {img_height}, width: {pixels.shape[1]}")
    print(f"Origin: {origin[:2]}, Resolution: {resolution}")
    print(f"Current world point (x_world, y_world): {current_point_world}")
    print(f"Current pixel point (x_pixel, y_pixel) = (row, col): ({current_point_pixel[0]:.1f}, {current_point_pixel[1]:.1f})")
    if 0 <= int(current_point_pixel[0]) < img_height and 0 <= int(current_point_pixel[1]) < pixels.shape[1]:
        pixel_color = pixels[int(current_point_pixel[0]), int(current_point_pixel[1])]
        print(f"Pixel color at that point: {pixel_color}")
    print()

    current_room = get_room_at_point(pixels, current_point_pixel, room_colors)

    if current_room:
        print(f"Current room: {current_room}")
        print()
        distances_to_neighbors(pixels, current_point_pixel, current_room, graph, room_colors, resolution)
    else:
        print(f"No room found at point {current_point_world}")

    # Visualize
    print()
    # visualize_graph(graph, config)
