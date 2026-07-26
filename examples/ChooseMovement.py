from collections import deque

from PIL import Image
import numpy as np
from scipy.ndimage import distance_transform_edt
import yaml
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import sdsl
from sdsl.loaders.load_pgm_map import load_pgm_map


class ChooseMovement:
    """Encapsulate room-edge and graph reasoning with optional voxel motion simulation."""

    def __init__(self, env=None, yaml_path=None, image_path=None):
        self.env = env
        self.config = self._load_map_config(yaml_path) if yaml_path else None
        self.pixels = self._load_image(image_path) if image_path else None
        self.resolution = self.config['resolution'] if self.config else None
        self.origin = tuple(self.config['origin'][:2]) if self.config and 'origin' in self.config else (0.0, 0.0)
        self.room_colors = {room['name']: room['rgb'] for room in self.config['rooms']} if self.config else {}
        self.img_height = self.pixels.shape[0] if self.pixels is not None else None
        self.room_graph = self.create_room_graph() if self.config else nx.Graph()
        self.room_distance_lookup = self.precompute_room_distance_lookup()

        self.wall_mask = np.all(self.pixels == (0, 0, 0), axis=2)
        self.distance_map = distance_transform_edt(~self.wall_mask)

    @staticmethod
    def _load_map_config(yaml_path):
        """Load map configuration from YAML."""
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def _load_image(image_path):
        """Load image as an RGB numpy array."""
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)

    def point_to_room_edge_distance(self, current_point, target_room):
        """
        Calculate distance from the current point to the nearest edge of a target room.

        Args:
            current_point: (x, y) in pixel coordinates where x is row, y is column
            target_room: target room name defined in the YAML config

        Returns:
            (distance_pixels, distance_meters) or (None, None) if out of bounds
        """
        target_room_rgb = self.room_colors[target_room]

        mask = np.all(self.pixels == target_room_rgb, axis=2)
        distance_map = distance_transform_edt(~mask)

        x_pixel, y_pixel = int(current_point[0]), int(current_point[1])
        h, w = distance_map.shape
        if not (0 <= x_pixel < h and 0 <= y_pixel < w):
            return None, None

        distance_pixels = distance_map[x_pixel, y_pixel]
        distance_meters = distance_pixels * self.resolution
        return distance_pixels, distance_meters

    def distance_to_closest_wall(self, current_point, wall_rgb=(0, 0, 0)):
        """
        Calculate distance from a point to the nearest wall pixel.

        Args:
            current_point: (x, y) in pixel coordinates where x is row, y is column
            wall_rgb: RGB tuple for the wall color (defaults to black)

        Returns:
            (distance_pixels, distance_meters) or (None, None) if unavailable.
        """
        # if self.pixels is None or self.resolution is None:
        #     return None, None

        # wall_mask = np.all(self.pixels == np.array(wall_rgb, dtype=np.uint8), axis=2)
        # if not np.any(wall_mask):
        #     return None, None

        distance_map = self.distance_map  # Precomputed distance transform to walls

        x_pixel, y_pixel = int(current_point[0]), int(current_point[1])
        h, w = distance_map.shape
        if not (0 <= x_pixel < h and 0 <= y_pixel < w):
            return None, None

        distance_pixels = float(distance_map[x_pixel, y_pixel])
        distance_meters = distance_pixels * self.resolution
        return distance_meters

    def room_to_room_distance(self, source_room, target_room):
        """
        Calculate the minimum distance between two room regions.

        Args:
            source_room: name of the source room defined in the YAML config
            target_room: name of the target room defined in the YAML config

        Returns:
            (distance_pixels, distance_meters) or (None, None) if either room is
            missing or the image/config is unavailable.
        """
        if self.pixels is None or self.resolution is None:
            return None, None

        if source_room not in self.room_colors or target_room not in self.room_colors:
            return None, None

        if source_room == target_room:
            return 0.0, 0.0

        source_rgb = self.room_colors[source_room]
        target_rgb = self.room_colors[target_room]

        source_mask = np.all(self.pixels == source_rgb, axis=2)
        target_mask = np.all(self.pixels == target_rgb, axis=2)

        if not np.any(source_mask) or not np.any(target_mask):
            return None, None

        distance_map = distance_transform_edt(~target_mask)
        source_distances = distance_map[source_mask]
        if source_distances.size == 0:
            return None, None

        distance_pixels = float(np.min(source_distances))
        distance_meters = distance_pixels * self.resolution
        return distance_pixels, distance_meters

    def get_room_at_point(self, current_point):
        """
        Determine which room a point is in based on pixel color.

        Args:
            current_point: (x, y) in pixel coordinates where x is row, y is column

        Returns:
            Room name or None if the point is out of bounds or unclassified.
        """
        x_pixel, y_pixel = int(current_point[0]), int(current_point[1])
        h, w = self.pixels.shape[:2]
        if not (0 <= x_pixel < h and 0 <= y_pixel < w):
            return None

        pixel_color = self.pixels[x_pixel, y_pixel]
        for room_name, room_rgb in self.room_colors.items():
            if np.array_equal(pixel_color, room_rgb):
                return room_name
        return None

    def world_to_pixel(self, world_point):
        """
        Convert world coordinates to pixel coordinates.

        Args:
            world_point: (world_x, world_y) where x is horizontal and y is vertical.

        Returns:
            (pixel_x, pixel_y) where pixel_x is row and pixel_y is column.
        """
        pixel_y = (world_point[0] - self.origin[0]) / self.resolution
        pixel_x = self.img_height - (world_point[1] - self.origin[1]) / self.resolution
        return int(round(pixel_x)), int(round(pixel_y))

    def pixel_to_world(self, pixel_point):
        """
        Convert pixel coordinates to world coordinates.

        Args:
            pixel_point: (pixel_x, pixel_y) where pixel_x is row and pixel_y is column.

        Returns:
            (world_x, world_y) where x is horizontal and y is vertical.
        """
        pixel_x, pixel_y = pixel_point
        world_x = pixel_y * self.resolution + self.origin[0]
        world_y = (self.img_height - pixel_x) * self.resolution + self.origin[1]
        return world_x, world_y

    def create_room_graph(self):
        """Create a graph from room configuration and initialize visit flags."""
        graph = nx.Graph()
        for room in self.config['rooms']:
            graph.add_node(room['name'], rgb=room['rgb'], visited=False)
        for edge in self.config.get('graph_edges', []):
            graph.add_edge(edge[0], edge[1])
        return graph

    def precompute_room_distance_lookup(self):
        """Precompute distances for room pairs whose graph distance is exactly 2."""
        if not self.room_graph:
            return {}

        if self.pixels is None or self.resolution is None:
            return {}

        lookup = {}
        for source_room, shortest_paths in nx.all_pairs_shortest_path_length(self.room_graph):
            for target_room, path_length in shortest_paths.items():
                if source_room == target_room or path_length != 2:
                    continue

                pair_key = tuple(sorted((source_room, target_room)))
                if pair_key in lookup:
                    continue

                distance = self.room_to_room_distance(source_room, target_room)
                if distance is not None:
                    lookup[pair_key] = distance

        return lookup

    def get_precomputed_room_distance(self, source_room, target_room):
        """Retrieve a cached room-to-room distance in O(1)."""
        if source_room == target_room:
            return 0.0, 0.0

        pair_key = tuple(sorted((source_room, target_room)))
        return self.room_distance_lookup.get(pair_key)

    def reset_visited_flags(self):
        """Reset the visited flag for every node in the room graph."""
        for node in self.room_graph.nodes:
            self.room_graph.nodes[node]['visited'] = False

    def mark_node_visited(self, room_name):
        """Mark a room node as visited."""
        if room_name in self.room_graph:
            self.room_graph.nodes[room_name]['visited'] = True

    @staticmethod
    def print_graph_info(graph):
        """Print graph statistics."""
        print(f"Rooms: {list(graph.nodes())}")
        print(f"Connections: {list(graph.edges())}")
        print(f"Number of rooms: {graph.number_of_nodes()}")
        print(f"Number of connections: {graph.number_of_edges()}")

    @staticmethod
    def find_path(graph, start_room, end_room):
        """Find path between two rooms."""
        try:
            return nx.shortest_path(graph, start_room, end_room)
        except nx.NetworkXNoPath:
            return None

    def _compute_neighbor_distances(self, current_point, graph, verbose=False):
        """
        Compute room-edge distances using breadth-first traversal over the room graph.

        The traversal starts from the current room. If the current room is unvisited,
        its distance is treated as 0. Otherwise, the BFS explores the current room's
        neighbors, then the neighbors of any visited room, while avoiding revisits and
        keeping the minimal distance discovered for each room.
        """
        current_room = self.get_room_at_point(current_point)
        if current_room is None:
            if verbose:
                print(f"  No room found at point {current_point}")
            return {}

        if current_room not in graph:
            if verbose:
                print(f"  {current_room} is not in graph")
            return {}

        if not graph.nodes[current_room].get('visited', False):
            if verbose:
                print(f"  Current room {current_room} is unvisited; distance = 0")
            return {current_room: 0.0}

        distances = {}
        queue = deque([(current_room, None, 0.0)])
        processed = set()

        if verbose:
            print(f"  BFS distances from point {current_point} starting at {current_room}:")

        while queue:
            room, parent, distance_from_parent = queue.popleft()
            if room in processed:
                continue
            processed.add(room)

            for neighbor in sorted(graph.neighbors(room)):
                if neighbor in processed:
                    continue

                if parent is None:
                    _, dist_m = self.point_to_room_edge_distance(current_point, neighbor)
                else:
                    _, dist_m = self.get_precomputed_room_distance(parent, neighbor)
                if dist_m is None:
                    if verbose:
                        print(f"    -> {neighbor}: out of bounds")
                    continue

                total_distance = distance_from_parent + dist_m
                

                if verbose:
                    print(f"    -> {neighbor}: {dist_m:.3f} m (total {total_distance:.3f} m)")

                if graph.nodes.get(neighbor, {}).get('visited', True): #continue BFS to its neighbors
                    queue.append((neighbor, room, total_distance))
                else: 
                    if neighbor not in distances or total_distance < distances[neighbor]:
                        distances[neighbor] = total_distance

        return distances

    def distances_to_neighbors(self, current_point, graph):
        """
        Calculate distance from the current point to the edge of each neighboring room.
        """
        return self._compute_neighbor_distances(current_point, graph, verbose=True)

    def min_distance_to_neighbor_rooms(self, current_point, graph):
        """
        Return the minimal distance to any neighboring room edge, or None if unavailable.
        """
        distances = self._compute_neighbor_distances(current_point, graph, verbose=False)
        neighbor_distances = [dist_m for dist_m in distances.values() if dist_m is not None]
        if not neighbor_distances:
            return None
        return min(neighbor_distances)

    @staticmethod
    def visualize_graph(graph, config, output_path='room_graph.png'):
        """Visualize the room connectivity graph."""
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(graph, seed=42, k=2)
        room_colors_dict = {room['name']: room['rgb'] for room in config['rooms']}
        node_colors = [[c / 255.0 for c in room_colors_dict[node]] for node in graph.nodes()]

        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=3000)
        nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold')
        nx.draw_networkx_edges(graph, pos, width=2, alpha=0.6)

        plt.title('Room Connectivity Graph')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Graph saved to {output_path}")

    @staticmethod
    def _translate_voxel(v, dx, dy):
        """Translate a Voxel_R3 in the XY plane."""
        bl = v.bottom_left
        tr = v.top_right
        bl_moved = sdsl.R3(bl[0] + dx, bl[1] + dy, bl[2])
        tr_moved = sdsl.R3(tr[0] + dx, tr[1] + dy, tr[2])
        return sdsl.Voxel_R3(bl_moved, tr_moved)

    @staticmethod
    def _voxel_midpoint(voxel):
        return voxel.midpoint()

    @staticmethod
    def _voxel_rectangle(voxel):
        bl = voxel.bottom_left
        tr = voxel.top_right
        width = tr[0] - bl[0]
        height = tr[1] - bl[1]
        return plt.Rectangle((bl[0], bl[1]), width, height)

    def visualize_voxels(self, ax, voxels, color='red', alpha=0.35, edgecolor='black', label=None):
        """Draw a list of 2D voxel projections on a Matplotlib axis."""
        if not voxels:
            return None
        if label is not None:
            for i, voxel in enumerate(voxels):
                mid = voxel.midpoint()
                # print(f"{label} voxel {i}: midpoint=({mid[0]:.6f}, {mid[1]:.6f}, {mid[2]:.6f})")
        patches = [self._voxel_rectangle(v) for v in voxels]
        coll = PatchCollection(patches, facecolor=color, edgecolor=edgecolor,
                       alpha=alpha, linewidths=0.5)
        ax.add_collection(coll)
        return coll
    
    def mark_voxels_visited(self, voxels):
        """Mark the rooms containing the given voxels as visited."""
        for voxel in voxels:
            mid = voxel.midpoint()
            pixel_point = self.world_to_pixel((float(mid[0]), float(mid[1])))
            room_name = self.get_room_at_point(pixel_point)
            if room_name is not None:
                self.mark_node_visited(room_name)
    
    def visualize_simulation_result(self, ax, simulation, base_voxels=None):
        """Visualize original voxels and the simulation's cleaned voxels.

        Args:
            ax: Matplotlib axes to draw the voxels onto.
            simulation: A simulation dictionary (or a list of them) that contains
                either 'cleaned_voxels' or 'surviving_voxels'.
            base_voxels: Optional original voxel set to draw first as the baseline.
        """
        for artist in list(ax.collections):
            if isinstance(artist, PatchCollection):
                artist.remove()

        if base_voxels is not None:
            self.visualize_voxels(
                ax,
                base_voxels,
                color='red',
                alpha=0.2,
                edgecolor='darkred',
                label='original',
            )

        simulations_to_plot = [simulation] if isinstance(simulation, dict) else list(simulation or [])
        for sim_idx, sim in enumerate(simulations_to_plot):
            voxels_to_plot = sim.get('cleaned_voxels') or sim.get('surviving_voxels') or []
            if not voxels_to_plot:
                continue

            self.visualize_voxels(
                ax,
                voxels_to_plot,
                color='orange',
                alpha=0.35,
                edgecolor='brown',
                label=f'sim_{sim_idx}',
            )

    def simulate_movements(self, angles, dists, voxels, beliefs, max_step=0.3):
        """Simulate moving voxels along each ray direction and evaluate collisions."""
        # angles = np.asarray(angles, dtype=float) #global
        angles = np.linspace(0, 2 * np.pi , 16, endpoint=False) #relative
        dists = np.asarray(dists, dtype=float)
        beliefs = np.asarray(beliefs, dtype=float)

        results = []
        if len(voxels) == 0 or len(angles) == 0:
            return results

        room_graph = self.room_graph
        self.mark_voxels_visited(voxels)
        for angle, dist in zip(angles, dists):
            step = float(np.minimum(np.maximum(dist-0.15, 0), max_step))
            if step <= 0.01: continue

            kept_voxels = []
            kept_beliefs = []
            for voxel, belief in zip(voxels, beliefs):
                alpha = (float(angle) + voxel.midpoint()[2]) % (2 * np.pi)  # global
                dx = step * np.cos(alpha)
                dy = step * np.sin(alpha) 
                moved_voxel = self._translate_voxel(voxel, dx, dy)
                orig_mid = self._voxel_midpoint(voxel)
                moved_mid = self._voxel_midpoint(moved_voxel)
                q_old = sdsl.R3(float(orig_mid[0]), float(orig_mid[1]), float(orig_mid[2]))
                q_new = sdsl.R3(float(moved_mid[0]), float(moved_mid[1]), float(moved_mid[2]))
                bottom_left = sdsl.R3(float(moved_voxel.bottom_left[0]), float(moved_voxel.bottom_left[1]), float(moved_voxel.bottom_left[2]))
                top_right = sdsl.R3(float(moved_voxel.top_right[0]), float(moved_voxel.top_right[1]), float(moved_voxel.top_right[2]))
                top_left = sdsl.R3(float(moved_voxel.bottom_left[0]), float(moved_voxel.top_right[1]), float(moved_voxel.midpoint()[2]))
                bottom_right = sdsl.R3(float(moved_voxel.top_right[0]), float(moved_voxel.bottom_left[1]), float(moved_voxel.midpoint()[2]))


                collided = (not self.env.contains(q_new)) or self.env.collision_detection(q_old, q_new)
                collided = collided or (not self.env.contains(bottom_left)) or (not self.env.contains(top_right)) or (not self.env.contains(top_left)) or (not self.env.contains(bottom_right))
                if not collided:
                    kept_voxels.append(moved_voxel)
                    kept_beliefs.append(float(belief))

            entropy = None
            avg_neighbor_room_distance = None
            # avg_wall_distance = None
            cleaned_voxels = []
            normalized_beliefs = []
            if kept_voxels:
                cleaned_voxels = sdsl.cleanup_SE2(kept_voxels)
                if cleaned_voxels:
                    entropy = float(sdsl.entropy_SE2(cleaned_voxels, list(kept_beliefs)))

                kept_beliefs = np.asarray(kept_beliefs, dtype=float)
                belief_sum = float(np.sum(kept_beliefs))
                norm_beliefs = kept_beliefs / belief_sum if belief_sum > 0 else np.full_like(kept_beliefs, 1.0 / len(kept_beliefs))
                normalized_beliefs = norm_beliefs.tolist()

                weighted_distance = 0.0
                valid_weight = 0.0
                weighted_wall_distance = 0.0
                for voxel, belief in zip(kept_voxels, norm_beliefs):
                    moved_mid = self._voxel_midpoint(voxel)
                    pixel_point = self.world_to_pixel((float(moved_mid[0]), float(moved_mid[1])))
                    current_room = self.get_room_at_point(pixel_point)
                    if current_room is None:
                        continue

                    min_dist = self.min_distance_to_neighbor_rooms(pixel_point, room_graph)
                    if min_dist is None:
                        continue
                    weighted_distance += belief * min_dist
                    
                    # min_wall_dist = self.distance_to_closest_wall(pixel_point)
                    # min_wall_dist = min(min_wall_dist, self.distance_to_closest_wall(self.world_to_pixel((float(voxel.top_right[0]), float(voxel.top_right[1])))))
                    # min_wall_dist = min(min_wall_dist, self.distance_to_closest_wall(self.world_to_pixel((float(voxel.top_right[0]), float(voxel.bottom_left[1])))))
                    # min_wall_dist = min(min_wall_dist, self.distance_to_closest_wall(self.world_to_pixel((float(voxel.bottom_left[0]), float(voxel.bottom_left[1])))))
                    # min_wall_dist = min(min_wall_dist, self.distance_to_closest_wall(self.world_to_pixel((float(voxel.bottom_left[0]), float(voxel.top_right[1])))))
                    
                    # weighted_wall_distance += belief * min_wall_dist
                    valid_weight += belief

                if valid_weight > 0:
                    avg_neighbor_room_distance = float(weighted_distance / valid_weight)
                    # avg_wall_distance = float(weighted_wall_distance / valid_weight)
                
            results.append({
                'angle': float(angle),
                'distance': float(dist),
                'step': step,
                'surviving_voxels': kept_voxels,
                'surviving_beliefs': kept_beliefs.tolist() if isinstance(kept_beliefs, np.ndarray) else kept_beliefs,
                'normalized_beliefs': normalized_beliefs,
                'cleaned_voxels': cleaned_voxels,
                'collision_count': len(voxels) - len(kept_voxels),
                'survival_ratio': len(kept_voxels) / len(voxels),
                'entropy': entropy,
                'avg_neighbor_room_distance': avg_neighbor_room_distance,
                # 'avg_wall_distance': avg_wall_distance,
            })

        return results

    def choose_movement(self, angles, dists, voxels, beliefs):
        """Choose the best movement based on simulated entropy."""
        simulation_results = self.simulate_movements(angles, dists, voxels, beliefs)
        if not simulation_results:
            return {'movement': None, 'reason': 'no_simulation_results', 'simulations': simulation_results}

        def score(sim):
                    entropy = sim.get('entropy')
                    avg_dist = sim.get('avg_neighbor_room_distance')
                    avg_wall_dist = sim.get('avg_wall_distance')
                    if entropy is None or avg_dist is None or avg_wall_dist is None:
                        return float('inf')
                    return entropy + avg_dist - avg_wall_dist

        best = min(
            simulation_results,
            key=score,
            default=None,
        )
        # best = min(
        #     simulation_results,
        #     key=lambda item: float('inf') if item['entropy'] is None else item['entropy'],
        # )

        return {
            'movement': {
                'type': 'step_along_angle',
                'angle': best['angle'],
                'dx': best['dx'],
                'dy': best['dy'],
                'distance': best['distance'],
                'step': best['step'],
            },
            'reason': 'min_entropy',
            'entropy': best['entropy'],
            'collision_count': best['collision_count'],
            'survival_ratio': best['survival_ratio'],
            # 'avg_wall_distance': best['avg_wall_distance'],
            'simulations': simulation_results,
        }


if __name__ == '__main__':
    yaml_path = 'resources/maps/2d/slam/simple_symmetry/symmetry_2.yaml'
    image_path = 'resources/maps/2d/slam/simple_symmetry/symmetry_2_c.png'

    pgm = load_pgm_map(yaml_path)
    env = sdsl.Env_2D_PGM(
        pgm.grid, pgm.resolution, pgm.origin_x, pgm.origin_y,
        pgm.occupied_thresh, pgm.negate,
    )
    bbox = env.bounding_box()

    mover = ChooseMovement(env, yaml_path, image_path)

    # Define a world point and convert to pixel coordinates
    current_point_world = (2.0, 2.12)  # world coordinates: (x_world, y_world) in meters
    current_point_pixel = mover.world_to_pixel(current_point_world)

    # Convert pixel coordinates to integer pixel indices used by the image arrays
    current_point = (int(round(current_point_pixel[0])), int(round(current_point_pixel[1])))

    # Find the room at the current pixel (if any)
    current_room = mover.get_room_at_point(current_point)

    # Choose a target room (pick any different room if available)
    all_rooms = list(mover.room_colors.keys())
    if not all_rooms:
        print('No rooms defined in YAML config')
        raise SystemExit(1)

    # call distances_to_neighbors
    if current_room is not None:
        print(f'Current room: {current_room}')
        mover.distances_to_neighbors(current_point, mover.room_graph)
    else:
        print(f'Current point {current_point} is not in any defined room') 

    # Example visualization of a simulated movement sweep
    z = 0.0
    angles = np.linspace(z, 2 * np.pi + z, 16, endpoint=False)
    dists = np.array([
        env.measure_distance(sdsl.R3(current_point_world[0], current_point_world[1], theta))
        for theta in angles
    ], dtype=float)

    odometry = [sdsl.R3(0.0, 0.0, theta - z) for theta in angles]
    pred = sdsl.Predicate_Fwd2D_Arr(env, odometry, list(dists), 1.0, 0.015)
    localized_voxels = sdsl.localize_omp_forkjoin_3d(
        bbox, pred, 10, timeout=1.0, verbose=False,
    )
    localized_voxels = sdsl.cleanup_SE2(localized_voxels)
    example_voxels = localized_voxels
    example_beliefs = [1.0 / len(example_voxels)] * len(example_voxels) if example_voxels else []

    sim_results = mover.simulate_movements(angles, dists, example_voxels, example_beliefs)
    print(f'Simulated {len(sim_results)} motion hypotheses')

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.subplots_adjust(right=0.88)
    ax.set_aspect('equal')
    ax.set_title('Voxel movement simulation')

    world_xmax = pgm.origin_x + pgm.width * pgm.resolution
    world_ymax = pgm.origin_y + pgm.height * pgm.resolution
    extent = [pgm.origin_x, world_xmax, pgm.origin_y, world_ymax]

    ax.imshow(
        pgm.grid,
        cmap='gray', origin='upper', extent=extent,
        interpolation='nearest', vmin=0, vmax=255,
        alpha=0.8, zorder=1,
    )
    ax.set_xlim(pgm.origin_x, world_xmax)
    ax.set_ylim(pgm.origin_y, world_ymax)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    # Plot current robot location similarly to the demo.
    current_world = mover.pixel_to_world(current_point)
    ax.plot([current_world[0]], [current_world[1]], marker='+', color='black',
            markersize=14, markeredgewidth=2, zorder=7)

    arrow_length = 0.5
    arrow_dx = arrow_length * np.cos(z)
    arrow_dy = arrow_length * np.sin(z)
    ax.arrow(
        current_world[0], current_world[1], arrow_dx, arrow_dy,
        head_width=0.1, head_length=0.1,
        fc='red', ec='red', linewidth=1.0, zorder=8)

    mover.visualize_simulation_result(
        ax,
        sim_results,
        base_voxels=example_voxels,
    )

    best_sim = min(
        (sim for sim in sim_results if sim['entropy'] is not None),
        key=lambda item: item['entropy'],
        default=None,
    )
    # if best_sim is not None:
    #     print('Best movement:', best_sim['angle'], best_sim['entropy'])
    #     mover.visualize_voxels(ax,
    #                            best_sim['cleaned_voxels'] or best_sim['surviving_voxels'],
    #                            color=(0.0, 1.0, 0.0, 0.35), edgecolor='green',
    #                            label='best')

    plt.show()
