import numpy as np
from scipy.spatial import Delaunay

from maie.common import Vec2


class PointOutsideTriangulationError(Exception):
    pass


class NearestNeighbors:
    def __init__(self, points):
        self.sol = Delaunay(points)

    def find_simplex_points(self, point: Vec2):
        res = self.sol.find_simplex(point)
        if res == -1:
            raise PointOutsideTriangulationError(point)

        simplex = self.sol.simplices[res]
        return [
            self.sol.points[i_point] for i_point in simplex
        ]

    def nearest_vertex(self, point: Vec2):
        return np.argmin(np.linalg.norm(self.sol.points - point, axis=1))

    def distance(self, i, j) -> float:
        pi = self.sol.points[i]
        pj = self.sol.points[j]
        return float(np.linalg.norm(pi - pj))

    def nearest_neighbors(self, point: Vec2, k: int) -> list[Vec2 | np.ndarray]:
        nv = self.nearest_vertex(point)
        ref = np.array(point)
        nearest = list(self.sol.points[i] for i in self._find_nearest_neighbors(nv, k))
        nearest.sort(key=lambda j: np.linalg.norm(j - ref))
        return nearest[:k]

    def _find_nearest_neighbors(self, i_point: int, k: int) -> set[int]:
        indeces, neighbors = self.sol.vertex_neighbor_vertices
        stack = [i_point]
        nearest = set()
        while stack:
            i = stack.pop()
            if i in nearest:
                continue

            nearest.add(int(i))
            if len(nearest) < k:
                start = indeces[i]
                stop = indeces[i + 1]
                new_points = neighbors[start:stop]
                for new_point in new_points:
                    stack.append(new_point)
        return nearest

    def __len__(self):
        return len(self.sol.points)
