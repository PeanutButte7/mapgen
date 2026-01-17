import math
from typing import TypeAlias

from maie.common import Vec2, Bounds, Segment, Size

OwnerId: TypeAlias = int
OwnerMap: TypeAlias = dict[Vec2, OwnerId]

def voronoi_partition_tiles(
        bounds: Bounds,
        sites: list[Vec2],
        tile_size: float,
) -> tuple[OwnerMap, Size, Vec2]:
    """
    Returns:
      owners[(tx, ty)] = index of nearest site
      (w_tiles, h_tiles) = grid size in tiles

    World->tile coords use:
      tx = floor((x - xmin)/tile_size), similarly for y.
    """
    xmin, ymin, xmax, ymax = bounds
    w = xmax - xmin
    h = ymax - ymin
    w_tiles = int(math.ceil(w / tile_size))
    h_tiles = int(math.ceil(h / tile_size))

    w_offset = int(math.ceil(xmin / tile_size))
    h_offset = int(math.ceil(ymin / tile_size))
    owners: OwnerMap = {}

    for ty in range(h_tiles):
        cy = ymin + (ty + 0.5) * tile_size
        for tx in range(w_tiles):
            cx = xmin + (tx + 0.5) * tile_size

            best_i = 0
            best_d2 = float("inf")
            for i, (sx, sy) in enumerate(sites):
                dx = cx - sx
                dy = cy - sy
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_i = i

            owners[(tx, ty)] = best_i

    return owners, (w_tiles, h_tiles), (w_offset, h_offset)


def voronoi_tile_borders(
        owners: OwnerMap,
        grid_size: Size,
) -> list[Segment]:
    """
    Returns a list of tile-edge segments (in tile coordinates) that lie on Voronoi borders.
    Each segment is ((tx0, ty0), (tx1, ty1)) where endpoints are on tile grid lines.
    You can draw these as world lines by converting to world coords * tile_size (+ xmin/ymin).
    """
    w_tiles, h_tiles = grid_size
    borders: list[Segment] = []

    for ty in range(h_tiles):
        for tx in range(w_tiles):
            a = owners[(tx, ty)]
            # right neighbor => vertical border
            if tx + 1 < w_tiles and owners[(tx + 1, ty)] != a:
                borders.append(((tx + 1, ty), (tx + 1, ty + 1)))

            # down neighbor => horizontal border
            if ty + 1 < h_tiles and owners[(tx, ty + 1)] != a:
                borders.append(((tx, ty + 1), (tx + 1, ty + 1)))

    return borders
