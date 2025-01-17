from typing import List

import numpy as np

from sdsl import R2xS1, Env_R2, DynamicObstacle
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


def visualize_simulation(env: Env_R2, q0: R2xS1, odometry: List[R2xS1], measurements: List[float], dynamic_obstacles: List[DynamicObstacle] = []):
    blue_segments = []
    red_segments = []
    points = []

    points.append([q0.x(), q0.y()])

    # Visualize rays
    for g, d in zip(odometry, measurements):
        source = g * q0
        source = [source.x(), source.y()]
        target = [source[0] + d * np.cos(q0.r() + g.r()), source[1] + d * np.sin(q0.r() + g.r())]
        points.append(source); points.append(target)
        outlier = is_outlier_measurement(env, g * q0, dynamic_obstacles)
        collection = red_segments if outlier else blue_segments
        collection.append([source[0], source[1], target[0], target[1]])
    
    visualize_2d(env, 
                dynamic_obstacles=dynamic_obstacles, 
                red_segments=np.array(red_segments), blue_segments=np.array(blue_segments), 
                points=np.array(points))
