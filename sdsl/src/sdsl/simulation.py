from typing import List, Tuple

import numpy as np

from sdsl import R2xS1, Env_R2, DynamicObstacle, DynamicObstacle_Disc2D, sample_q0
from sdsl.visualization.viz2d import visualize_2d, voxel_to_segments


def measure_distance(env: Env_R2, q: R2xS1, dynamic_obstacles: List[DynamicObstacle]) -> float:
    dist = env.measure_distance(q)
    for obs in dynamic_obstacles:
        dist = min(dist, obs.measureDistance(q))
    return dist

def is_outlier_measurement(env: Env_R2, q: R2xS1, dynamic_obstacles: List[DynamicObstacle]) -> bool:
    dist = env.measure_distance(q)
    for obs in dynamic_obstacles:
        if obs.measureDistance(q) < dist:
            return True
    return False

def get_odometry(k: int, sensor_offset: float) -> List[R2xS1]:
    odometry = []
    for i in range(k):
        angle = 2.0 * np.pi * i / k
        odometry.append(R2xS1(sensor_offset * np.cos(angle), sensor_offset * np.sin(angle), angle))
    return odometry

def get_measurements(env: Env_R2, odometry: List[R2xS1], q0: R2xS1, dynamic_obstacles: List[DynamicObstacle]) -> List[float]:
    measurements = []
    for g in odometry:
        measurements.append(measure_distance(env, g * q0, dynamic_obstacles))
    return measurements

def sample_uniform_dynamic_obstacles(env: Env_R2, n: int, radius: float) -> List[DynamicObstacle]:
    obstacles = []
    for _ in range(n):
        q0 = sample_q0(env)
        obstacles.append(DynamicObstacle_Disc2D(q0.x(), q0.y(), radius))
    return obstacles

def verify_scenario(env: Env_R2, dynamic_obstacles: List[DynamicObstacle], q0: R2xS1, odometry: List[R2xS1]) -> bool:
    if not env.is_inside(q0):
        return False
    for g in odometry:
        q = g * q0
        if not env.is_inside(q):
            return False
    for obs in dynamic_obstacles:
        o = R2xS1(obs.x, obs.y, 0.0)
        if not env.is_inside(o):
            return False
        if env.hausdorff_distance(o) <= obs.radius:
            return False
    for g in odometry:
        q = g * q0
        for obs in dynamic_obstacles:
            if (q.x() - obs.x)**2 + (q.y() - obs.y)**2 <= obs.radius**2:
                return False
    return True

def get_valid_scenario(env: Env_R2, k: int, sensor_offset: float, n: int, radius: float) -> Tuple[R2xS1, List[R2xS1], List[DynamicObstacle]]:
    num_tries = 0
    while True:
        q0 = sample_q0(env)
        odometry = get_odometry(k, sensor_offset)
        dynamic_obstacles = sample_uniform_dynamic_obstacles(env, n, radius)
        if verify_scenario(env, dynamic_obstacles, q0, odometry):
            return q0, odometry, dynamic_obstacles
        num_tries += 1
        if num_tries > 10:
            return None, None, None


def visualize_simulation(env: Env_R2, q0: R2xS1, odometry: List[R2xS1], measurements: List[float], dynamic_obstacles: List[DynamicObstacle] = []):
    blue_segments = []
    red_segments = []
    magenta_segments = []
    points = []
    point_colors = []

    points.append([q0.x(), q0.y()])
    point_colors.append("#00ff00")

    # Visualize rays
    for g, d in zip(odometry, measurements):
        source = g * q0
        source = [source.x(), source.y()]
        target = [source[0] + d * np.cos(q0.r() + g.r()), source[1] + d * np.sin(q0.r() + g.r())]
        points.append(source); point_colors.append("#00ff00")
        points.append(target); point_colors.append("magenta")
        outlier = is_outlier_measurement(env, g * q0, dynamic_obstacles)
        collection = red_segments if outlier else magenta_segments
        collection.append([source[0], source[1], target[0], target[1]])
    
    visualize_2d(env, 
                dynamic_obstacles=dynamic_obstacles, 
                red_segments=np.array(red_segments), blue_segments=np.array(blue_segments), magenta_segments=np.array(magenta_segments),
                points=np.array(points), point_colors=point_colors)