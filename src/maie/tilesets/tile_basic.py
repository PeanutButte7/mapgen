
from enum import IntEnum

import pygame


class TileType(IntEnum):
    GRASS = 0
    SAND = 1
    WATER = 2
    PATH = 3


TileColors = {
    TileType.GRASS: (40, 220, 30),  # GREEN
    TileType.SAND: (255, 255, 40),  # YELLOW
    TileType.WATER: (40, 120, 255),  # BLUE
}

allowed_neighbors = {
    TileType.GRASS: [TileType.GRASS, TileType.SAND, TileType.PATH],
    TileType.SAND: [TileType.SAND, TileType.GRASS, TileType.WATER],
    TileType.WATER: [TileType.WATER, TileType.SAND],
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
            TileType.GRASS: 0.50,
            TileType.SAND: 0.25,
            TileType.WATER: 0.25,
        }

    def all_ids(self) -> list[TileType]:
        return [t.type for t in self.tiles]

    @property
    def weights(self):
        return self._weights

    def is_allowed_neighbor(self, root: TileType, neighbor: TileType, direction) -> bool:
        return root in allowed_neighbors[neighbor]

    def draw_tile(self, tile_id: TileType, surf, x, y, scale=1.0):
        self._by_type[tile_id].draw(surf, x, y, scale)

    def __len__(self):
        return len(self.tiles)


def get_tileset(size):
    return SimpleTileSet([
        Tile(TileType.GRASS, size),
        Tile(TileType.SAND, size),
        Tile(TileType.WATER, size),
    ])