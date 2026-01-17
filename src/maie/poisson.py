import math
import random

from maie.common import Bounds, Vec2


from typing import Optional

def poisson_disc_2d(
    bounds: Bounds,
    radius: float,
    n_points: int,
    k: int = 30,
    seed: Optional[int] = None,
) -> list[Vec2]:
    """
    Poisson disc sampling in 2D using Bridson's algorithm.

    bounds   : (xmin, ymin, xmax, ymax)
    radius   : minimum distance between points
    n_points : target number of points (best effort)
    k        : attempts per active point
    seed     : RNG seed (optional)

    returns: list of (x, y)
    """
    if seed is not None:
        random.seed(seed)

    xmin, ymin, xmax, ymax = bounds
    w = xmax - xmin
    h = ymax - ymin

    cell_size = radius / math.sqrt(2)
    grid_w = int(math.ceil(w / cell_size))
    grid_h = int(math.ceil(h / cell_size))

    grid: list[Optional[int]] = [
        [None] * grid_h for _ in range(grid_w)
    ]

    def grid_coords(p: Vec2) -> Vec2:
        return (
            int((p[0] - xmin) / cell_size),
            int((p[1] - ymin) / cell_size),
        )

    def in_bounds(p: Vec2) -> bool:
        return xmin <= p[0] < xmax and ymin <= p[1] < ymax

    def far_enough(p: Vec2) -> bool:
        gx, gy = grid_coords(p)
        r2 = radius * radius

        for i in range(max(0, gx - 2), min(grid_w, gx + 3)):
            for j in range(max(0, gy - 2), min(grid_h, gy + 3)):
                idx = grid[i][j]
                if idx is not None:
                    qx, qy = points[idx]
                    dx = p[0] - qx
                    dy = p[1] - qy
                    if dx * dx + dy * dy < r2:
                        return False
        return True

    # --- init ---
    x0 = random.uniform(xmin, xmax)
    y0 = random.uniform(ymin, ymax)

    points: list[Vec2] = [(x0, y0)]
    active: list[int] = [0]

    gx, gy = grid_coords(points[0])
    grid[gx][gy] = 0

    # --- main loop ---
    while active and len(points) < n_points:
        idx = random.choice(active)
        base = points[idx]
        found = False

        for _ in range(k):
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(radius, 2 * radius)
            p = (
                base[0] + r * math.cos(angle),
                base[1] + r * math.sin(angle),
            )

            if in_bounds(p) and far_enough(p):
                points.append(p)
                active.append(len(points) - 1)
                gx, gy = grid_coords(p)
                grid[gx][gy] = len(points) - 1
                found = True
                break

        if not found:
            active.remove(idx)

    return points
