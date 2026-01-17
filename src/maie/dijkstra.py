import heapq
from typing import Callable, TypeAlias

import numpy as np
import numpy.typing as npt

from maie.common import GVec2

N4 = [(1,0), (-1,0), (0,1), (0,-1)]
N8 = N4 + [(1,1), (1,-1), (-1,1), (-1,-1)]

Path: TypeAlias = list[GVec2]

def in_bounds(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height

def dijkstra_grid(
    width: int,
    height: int,
    passable: Callable[[int, int], bool], # function (x,y) -> bool  OR set of blocked cells you check via (x,y) not in blocked
    cost_of: Callable[[int, int, int, int], float], # (edge cost)
    start: GVec2,
    goal = GVec2,
    use_diagonals=False
) -> tuple[npt.NDArray, npt.NDArray[GVec2]]:
    neigh = N8 if use_diagonals else N4

    dist = np.full((height, width), fill_value=np.inf)
    parent = np.full((height, width), fill_value=None)

    sx, sy = start
    dist[sy][sx] = 0.0
    pq = [(0.0, sx, sy)]

    while pq:
        d, x, y = heapq.heappop(pq)
        if d != dist[y][x]:
            continue

        # terminate if goal reached
        if goal and (x, y) == goal:
            break

        # iterate through neighbourhood
        for dx, dy in neigh:
            nx, ny = x + dx, y + dy

            if not in_bounds(nx, ny, width, height):
                continue
            if not passable(nx, ny):
                continue

            c = cost_of(x, y, nx, ny)
            nd = d + c
            if nd < dist[ny][nx]:
                dist[ny][nx] = nd
                parent[ny][nx] = (x, y)
                heapq.heappush(pq, (nd, nx, ny))

    return dist, parent


def reconstruct_path_grid(
        parent: npt.NDArray[GVec2],
        start: GVec2,
        goal: GVec2
) -> list[GVec2]:
    path = []
    curr = goal
    while curr:
        path.append(curr)
        if curr == start:
            break
        curr = parent[curr[1], curr[0]]
    path.reverse()
    return path
