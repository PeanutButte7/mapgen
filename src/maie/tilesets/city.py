from enum import IntEnum
from typing import Dict, Set

import pygame

from maie.common import NeighborIndex


class TileType(IntEnum):
    # background tiles
    EMPTY = 0  # empty lot / low-density
    BUILDING = 1  # dense block

    # road tiles
    ROAD_H = 2  # horizontal
    ROAD_V = 3  # vertical
    ROAD_CORNER_NE = 4
    ROAD_CORNER_NW = 5
    ROAD_CORNER_SE = 6
    ROAD_CORNER_SW = 7
    ROAD_T_N = 8  # T-junction, open to the North
    ROAD_T_E = 9
    ROAD_T_S = 10
    ROAD_T_W = 11
    ROAD_X = 12  # 4-way intersection


# simple colors; you can later switch to loading proper PNGs
BACKGROUND_COLOR = (40, 40, 40)  # default fill
EMPTY_COLOR = (50, 120, 50)
BUILDING_COLOR = (120, 120, 150)
ROAD_COLOR = (200, 200, 200)
PARK_COLOR = (40, 100, 40)  # unused here, but you can add a PARK tile later


# non-directional compatibility (which *types* can touch at all)
# we’ll enforce road connections separately using path_connections.
allowed_neighbors: Dict[TileType, Set[TileType]] = {
    TileType.EMPTY: {
        TileType.EMPTY, TileType.BUILDING,
        TileType.ROAD_H, TileType.ROAD_V,
        TileType.ROAD_CORNER_NE, TileType.ROAD_CORNER_NW,
        TileType.ROAD_CORNER_SE, TileType.ROAD_CORNER_SW,
        TileType.ROAD_T_N, TileType.ROAD_T_E,
        TileType.ROAD_T_S, TileType.ROAD_T_W,
        TileType.ROAD_X,
    },
    TileType.BUILDING: {
        TileType.EMPTY, TileType.BUILDING,
        TileType.ROAD_H, TileType.ROAD_V,
        TileType.ROAD_CORNER_NE, TileType.ROAD_CORNER_NW,
        TileType.ROAD_CORNER_SE, TileType.ROAD_CORNER_SW,
        TileType.ROAD_T_N, TileType.ROAD_T_E,
        TileType.ROAD_T_S, TileType.ROAD_T_W,
        TileType.ROAD_X,
    },
}

# roads are allowed next to background and other roads
road_tiles = [
    TileType.ROAD_H, TileType.ROAD_V,
    TileType.ROAD_CORNER_NE, TileType.ROAD_CORNER_NW,
    TileType.ROAD_CORNER_SE, TileType.ROAD_CORNER_SW,
    TileType.ROAD_T_N, TileType.ROAD_T_E,
    TileType.ROAD_T_S, TileType.ROAD_T_W,
    TileType.ROAD_X,
]

for rt in road_tiles:
    allowed_neighbors[rt] = {
        TileType.EMPTY, TileType.BUILDING,
        *road_tiles,
    }


# directional “road presence” on each edge
path_connections: Dict[TileType, Set[NeighborIndex]] = {
    TileType.ROAD_H: {NeighborIndex.LEFT, NeighborIndex.RIGHT},
    TileType.ROAD_V: {NeighborIndex.UP, NeighborIndex.DOWN},

    TileType.ROAD_CORNER_NE: {NeighborIndex.UP, NeighborIndex.RIGHT},
    TileType.ROAD_CORNER_NW: {NeighborIndex.UP, NeighborIndex.LEFT},
    TileType.ROAD_CORNER_SE: {NeighborIndex.DOWN, NeighborIndex.RIGHT},
    TileType.ROAD_CORNER_SW: {NeighborIndex.DOWN, NeighborIndex.LEFT},

    TileType.ROAD_T_N: {NeighborIndex.LEFT, NeighborIndex.RIGHT, NeighborIndex.UP},
    TileType.ROAD_T_E: {NeighborIndex.UP, NeighborIndex.DOWN, NeighborIndex.RIGHT},
    TileType.ROAD_T_S: {NeighborIndex.LEFT, NeighborIndex.RIGHT, NeighborIndex.DOWN},
    TileType.ROAD_T_W: {NeighborIndex.UP, NeighborIndex.DOWN, NeighborIndex.LEFT},

    TileType.ROAD_X: {
        NeighborIndex.UP, NeighborIndex.DOWN,
        NeighborIndex.LEFT, NeighborIndex.RIGHT,
    },
    # EMPTY / BUILDING: no connections → no roads “inside”
}


OPPOSITE = {
    NeighborIndex.UP: NeighborIndex.DOWN,
    NeighborIndex.DOWN: NeighborIndex.UP,
    NeighborIndex.LEFT: NeighborIndex.RIGHT,
    NeighborIndex.RIGHT: NeighborIndex.LEFT,
}


def draw_base(s, w, h):
    road_w = max(4, w // 3)
    m_w, m_h = w - road_w, h - road_w
    draw_horiz(s, w, h, road_w)
    draw_vert(s, w, h, road_w)
    return road_w, m_w, m_h


def draw_vert(s, w, h, road_w):
    x0 = (w - road_w) // 2
    pygame.draw.rect(s, ROAD_COLOR, (x0, 0, road_w, h))


def draw_horiz(s, w, h, road_w):
    y0 = (h - road_w) // 2
    pygame.draw.rect(s, ROAD_COLOR, (0, y0, w, road_w))


def draw_empty(s, w, h):
    s.fill(EMPTY_COLOR)


def draw_buliding(s, w, h):
    s.fill(BUILDING_COLOR)
    for x in range(4, w, 6):
        for y in range(4, h, 6):
            pygame.draw.rect(s, (220, 220, 100), (x, y, 2, 2))


def draw_road_h(s, w, h):
    road_w = max(4, w // 3)
    draw_horiz(s, w, h, road_w)


def draw_road_v(s, w, h):
    road_w = max(4, w // 3)
    draw_vert(s, w, h, road_w)


def draw_corner_ne(s, w, h):
    road_w, m_w, m_h = draw_base(s, w, h)
    pygame.draw.rect(s, BACKGROUND_COLOR, (0, 0, m_w // 2, h))
    pygame.draw.rect(s, BACKGROUND_COLOR, (0, h - m_h // 2, w, m_h // 2))


def draw_corner_nw(s, w, h):
    road_w, m_w, m_h = draw_base(s, w, h)
    pygame.draw.rect(s, BACKGROUND_COLOR, (w - m_w // 2, 0, m_w // 2, h))
    pygame.draw.rect(s, BACKGROUND_COLOR, (0, h - m_h // 2, w, m_h // 2))


def draw_corner_se(s, w, h):
    road_w, m_w, m_h = draw_base(s, w, h)
    pygame.draw.rect(s, BACKGROUND_COLOR, (0, 0, m_w // 2, h))
    pygame.draw.rect(s, BACKGROUND_COLOR, (0, 0, w, m_h // 2))


def draw_corner_sw(s, w, h):
    road_w, m_w, m_h = draw_base(s, w, h)
    pygame.draw.rect(s, BACKGROUND_COLOR, (0, 0, w, m_h // 2))
    pygame.draw.rect(s, BACKGROUND_COLOR, (w - m_w // 2, 0, m_w // 2, h))


def draw_t_n(s, w, h):
    road_w, m_w, m_h = draw_base(s, w, h)
    pygame.draw.rect(s, BACKGROUND_COLOR, (0, h - m_h // 2, w, m_h // 2))


def draw_t_s(s, w, h):
    road_w, m_w, m_h = draw_base(s, w, h)
    pygame.draw.rect(s, BACKGROUND_COLOR, (0, 0, w, m_h // 2))


def draw_t_e(s, w, h):
    road_w, m_w, m_h = draw_base(s, w, h)
    pygame.draw.rect(s, BACKGROUND_COLOR, (0, 0, m_w // 2, h))


def draw_t_w(s, w, h):
    road_w, m_w, m_h = draw_base(s, w, h)
    pygame.draw.rect(s, BACKGROUND_COLOR, (h - m_w // 2, 0, m_w // 2, h))


def draw_x(s, w, h):
    draw_base(s, w, h)


drawers = {
    TileType.EMPTY: draw_empty,
    TileType.BUILDING: draw_buliding,
    TileType.ROAD_H: draw_road_h,
    TileType.ROAD_V: draw_road_v,
    TileType.ROAD_CORNER_NE: draw_corner_ne,
    TileType.ROAD_CORNER_SE: draw_corner_se,
    TileType.ROAD_CORNER_SW: draw_corner_sw,
    TileType.ROAD_CORNER_NW: draw_corner_nw,
    TileType.ROAD_T_N: draw_t_n,
    TileType.ROAD_T_S: draw_t_s,
    TileType.ROAD_T_E: draw_t_e,
    TileType.ROAD_T_W: draw_t_w,
    TileType.ROAD_X: draw_x,
}


class Tile:
    def __init__(self, type: TileType, size: int = 24):
        self.type = type
        self.size = size
        self.surf = pygame.Surface((size, size), pygame.SRCALPHA)
        self._draw()

    def _draw(self):
        s = self.surf
        s.fill(BACKGROUND_COLOR)

        drawers[self.type](self.surf, self.size, self.size)

    def draw(self, surf, x, y, scale=1.0):
        size = (self.size*scale, self.size*scale)
        surf.blit(pygame.transform.scale(self.surf, size), (x, y))


class CityTileSet:
    """
    Implements the TileSetLike protocol from your core.WFCTile:
      - draw_tile(i, screen, x, y)
      - __len__()
      - is_allowed_neighbor(root, neighbor, direction)
    Tile indices are the same as TileType values.
    """

    def __init__(self, size: int = 24):
        self.tiles = [Tile(tt, size) for tt in TileType]

        # default weights (you can tweak)
        self.weights = {
            TileType.EMPTY: 0.15,
            TileType.BUILDING: 0.35,
            TileType.ROAD_H: 0.10,
            TileType.ROAD_V: 0.10,
            TileType.ROAD_CORNER_NE: 0.05,
            TileType.ROAD_CORNER_NW: 0.05,
            TileType.ROAD_CORNER_SE: 0.05,
            TileType.ROAD_CORNER_SW: 0.05,
            TileType.ROAD_T_N: 0.03,
            TileType.ROAD_T_E: 0.03,
            TileType.ROAD_T_S: 0.03,
            TileType.ROAD_T_W: 0.03,
            TileType.ROAD_X: 0.03,
        }

    def all_ids(self) -> list[TileType]:
        return [t.type for t in self.tiles]

    def draw_tile(self, i, surf, x, y, scale=1.0):
        self.tiles[i].draw(surf, x, y, scale)

    def __len__(self):
        return len(self.tiles)

    def is_allowed_neighbor(self, root, neighbor, direction: NeighborIndex) -> bool:
        root_t = TileType(root)
        neigh_t = TileType(neighbor)

        # 1) basic compatibility (roads next to roads / blocks; blocks next to anything)
        if root_t not in allowed_neighbors.get(neigh_t, set()):
            return False

        # 2) road connectivity: enforce that road “stubs” match
        cr = path_connections.get(root_t, set())
        cn = path_connections.get(neigh_t, set())
        opp = OPPOSITE[direction]

        root_has = direction in cr
        neigh_has = opp in cn

        # if exactly one side has a road, forbid (no dead half-connections)
        if root_has != neigh_has:
            return False

        return True


def get_tileset(size: int = 24):
    """
    Factory function matching your other tilesets.
    """
    return CityTileSet(size)
