import math

import shapely

INF = 10000 # Arbitrary large number, like "infinity"


class DynamicObstacle:
    def measureDistance(self, q):
        raise NotImplementedError()
    
class DynamicObstacle_Disc2D(DynamicObstacle):
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
    
    def measureDistance(self, q):
        circle = shapely.geometry.Point(self.x, self.y).buffer(self.radius).boundary
        ray = shapely.geometry.LineString([
            (q.x(), q.y()), 
            (q.x() + INF * math.cos(q.r()), q.y() + INF * math.sin(q.r()))
        ])
        isect = circle.intersection(ray)
        if isect.is_empty:
            return math.inf
        d = math.inf
        for p in isect.geoms:
            p = p.coords[0]
            d = min(d, (p[0] - q.x()) ** 2 + (p[1] - q.y()) ** 2)
        return math.sqrt(d)