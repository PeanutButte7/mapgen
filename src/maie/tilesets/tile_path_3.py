from enum import IntEnum

import pygame

from maie.common import NeighborIndex  # <-- use directions from common.py

DARK_GRAY = (40, 40, 40)
BLUE = (40, 120, 255)
YELLOW = (255, 255, 40)
GREEN = (40, 220, 30)

class TileType(IntEnum):
    GRASS = 0
    SAND = 1
    WATER = 2

    # oriented path pieces
    PATH_STRAIGHT_H = 3  # connects LEFT <-> RIGHT
    PATH_STRAIGHT_V = 4  # connects UP <-> DOWN
    PATH_CORNER_NE = 5   # connects UP <-> RIGHT
    PATH_CORNER_NW = 6   # connects UP <-> LEFT
    PATH_CORNER_SE = 7   # connects DOWN <-> RIGHT
    PATH_CORNER_SW = 8   # connects DOWN <-> LEFT


TileColors = {
    TileType.GRASS: GREEN,
    TileType.SAND: YELLOW,
    TileType.WATER: BLUE,

    TileType.PATH_STRAIGHT_H: DARK_GRAY,
    TileType.PATH_STRAIGHT_V: DARK_GRAY,
    TileType.PATH_CORNER_NE: DARK_GRAY,
    TileType.PATH_CORNER_NW: DARK_GRAY,
    TileType.PATH_CORNER_SE: DARK_GRAY,
    TileType.PATH_CORNER_SW: DARK_GRAY,
}

PATH_TILES = [
    TileType.PATH_STRAIGHT_H,
    TileType.PATH_STRAIGHT_V,
    TileType.PATH_CORNER_NE,
    TileType.PATH_CORNER_NW,
    TileType.PATH_CORNER_SE,
    TileType.PATH_CORNER_SW,
]

class Biome(IntEnum):
    WATER = 0
    SHORE = 1
    LAND = 2

tile_biome: dict[TileType, Biome] = {
    TileType.GRASS: Biome.LAND,
    TileType.SAND:  Biome.SHORE,
    TileType.WATER: Biome.WATER,
} | {t: Biome.LAND for t in PATH_TILES}

allowed_neighbors = {
    TileType.GRASS: [TileType.GRASS, TileType.SAND, *PATH_TILES],
    TileType.SAND: [TileType.SAND, TileType.GRASS, TileType.WATER, *PATH_TILES],
    TileType.WATER: [TileType.WATER, TileType.SAND],
}

biome_neighbors: dict[Biome, set[Biome]] = {
    Biome.WATER: {Biome.WATER, Biome.SHORE},
    Biome.SHORE: {Biome.WATER, Biome.SHORE, Biome.LAND},
    Biome.LAND:  {Biome.SHORE, Biome.LAND},
}

for p in PATH_TILES:
    allowed_neighbors[p] = [TileType.GRASS, TileType.SAND, *PATH_TILES]


path_connections: dict[TileType, set[NeighborIndex]] = {
    TileType.PATH_STRAIGHT_H: {NeighborIndex.LEFT, NeighborIndex.RIGHT},
    TileType.PATH_STRAIGHT_V: {NeighborIndex.UP, NeighborIndex.DOWN},

    TileType.PATH_CORNER_NE:  {NeighborIndex.UP,   NeighborIndex.RIGHT},
    TileType.PATH_CORNER_NW:  {NeighborIndex.UP,   NeighborIndex.LEFT},
    TileType.PATH_CORNER_SE:  {NeighborIndex.DOWN, NeighborIndex.RIGHT},
    TileType.PATH_CORNER_SW:  {NeighborIndex.DOWN, NeighborIndex.LEFT},
}


OPPOSITE = {
    NeighborIndex.UP: NeighborIndex.DOWN,
    NeighborIndex.DOWN: NeighborIndex.UP,
    NeighborIndex.LEFT: NeighborIndex.RIGHT,
    NeighborIndex.RIGHT: NeighborIndex.LEFT,
}


class Tile:
    def __init__(self, type: TileType, size=24):
        self.type = type
        self.surf = pygame.Surface((size, size))
        self.surf.fill(TileColors[self.type])
        self.size = size

    def draw(self, surf, x, y, scale=1.0):
        size = (self.size*scale, self.size*scale)
        surf.blit(pygame.transform.scale(self.surf, size), (x, y))


class SimpleTileSet:
    def __init__(self, base_tiles):
        self.tiles: list[Tile] = base_tiles
        self._by_type = {t.type: t for t in base_tiles}
        self._weights = {
            TileType.GRASS: 0.40,  # GRASS
            TileType.SAND: 0.10,  # SAND
            TileType.WATER: 0.35,  # WATER
            TileType.PATH_STRAIGHT_H: 0.02,  # PATH_STRAIGHT_H
            TileType.PATH_STRAIGHT_V: 0.02,  # PATH_STRAIGHT_V
            TileType.PATH_CORNER_NE: 0.02,  # PATH_CORNER_NE
            TileType.PATH_CORNER_NW: 0.02,  # PATH_CORNER_NW
            TileType.PATH_CORNER_SE: 0.01,  # PATH_CORNER_SE
            TileType.PATH_CORNER_SW: 0.01,  # PATH_CORNER_SW
        }

    def all_ids(self) -> list[TileType]:
        return [t.type for t in self.tiles]

    @property
    def weights(self):
        return self._weights

    def is_allowed_neighbor(self, root: int, neighbor: int, direction: NeighborIndex) -> bool:
        root_t = TileType(root)
        neigh_t = TileType(neighbor)

        # 1) terrain-level compatibility (rough "biome" control)
        if root_t not in allowed_neighbors.get(neigh_t, []):
            return False

        # 2) path connectivity: avoid half-connections and isolated path pixels
        cr = path_connections.get(root_t, set())
        cn = path_connections.get(neigh_t, set())
        opp = OPPOSITE[direction]

        root_has = direction in cr
        neigh_has = opp in cn

        if root_has != neigh_has:
            return False

        br = tile_biome[root_t]
        bn = tile_biome[neigh_t]
        if bn not in biome_neighbors[br]:
            return False

        return True

    def draw_tile(self, tile_id: TileType, surf, x, y, scale=1.0):
        self._by_type[tile_id].draw(surf, x, y)

    def __len__(self):
        return len(self.tiles)


def get_tileset(size=24) -> SimpleTileSet:
    base_tiles = [
        Tile(TileType.GRASS, size),
        Tile(TileType.SAND, size),
        Tile(TileType.WATER, size),
        Tile(TileType.PATH_STRAIGHT_H, size),
        Tile(TileType.PATH_STRAIGHT_V, size),
        Tile(TileType.PATH_CORNER_NE, size),
        Tile(TileType.PATH_CORNER_NW, size),
        Tile(TileType.PATH_CORNER_SE, size),
        Tile(TileType.PATH_CORNER_SW, size),
    ]
    return SimpleTileSet(base_tiles)
