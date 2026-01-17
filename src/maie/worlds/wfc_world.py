import importlib
from dataclasses import dataclass
from typing import Iterable

import math
import numpy as np
import pygame

import maie.tilesets as tilesets
from maie.camera import RenderContext, DrawLayer
from maie.common import GVec2
from maie.wavefunctioncollapse import WFCTile, TileSetLike

COLOR_WORLD_EDGE = (150, 0, 0)

def load_tileset(name: str, cell_size) -> TileSetLike:
    module_name = f"{tilesets.__name__}.{name}"
    mod = importlib.import_module(module_name)
    return mod.get_tileset(cell_size)


@dataclass
class WfcWorldConfig:
    width: int = 512
    height: int = 512
    tile_size: float = 16.
    seed: int = 6
    tileset: str = "city"

class WfcWorld:
    def __init__(self, cfg: WfcWorldConfig):
        self.width = cfg.width
        self.height = cfg.height
        self.ts = cfg.tile_size
        self.cfg = cfg

        self._regenerate()

    def get_layers(self, which: Iterable[int]) -> list[DrawLayer]:
        return [
            DrawLayer(z=1, label="world_border", draw=self._draw_world_border),
        ]

    def debug_layers(self):
        return [
            DrawLayer(z=30, label="ridge", draw=self._draw_wfc),
        ]

    def _draw_world_border(self, ctx: RenderContext) -> None:
        w, h = self.width, self.height
        origin = ctx.camera.world_to_screen((0, 0))
        corner = ctx.camera.world_to_screen((w, h))
        pygame.draw.line(ctx.screen, COLOR_WORLD_EDGE, origin, (corner[0], origin[1]), 3)  # x-axis
        pygame.draw.line(ctx.screen, COLOR_WORLD_EDGE, origin, (origin[0], corner[1]), 3)  # x-axis
        pygame.draw.line(ctx.screen, COLOR_WORLD_EDGE, corner, (origin[0], corner[1]), 3)  # x-axis
        pygame.draw.line(ctx.screen, COLOR_WORLD_EDGE, corner, (corner[0], origin[1]), 3)  # x-axis

    @property
    def shape_tiles(self):
        return int(math.floor(self.width / self.ts)), int(math.floor(self.height / self.ts))

    def is_tile_in_bounds(self, tile: GVec2) -> bool:
        tw, th = self.shape_tiles
        x, y = tile
        return 0 <= x < tw and 0 <= y < th

    def _draw_wfc(self, ctx: RenderContext) -> None:
        for tile, val in np.ndenumerate(self.wfc):
            tt = next(iter(val))
            pos = (tile[1] * self.ts, tile[0] * self.ts)
            pos = ctx.camera.world_to_screen(pos)
            self.tileset.draw_tile(tt, ctx.screen, *pos, ctx.camera.zoom)

    def _regenerate(self):
        self._generate_wfc()

    def _generate_wfc(self):
        wx, wy = self.shape_tiles
        self.tileset = load_tileset(self.cfg.tileset, int(self.ts))
        wfccore = WFCTile(self.tileset, wx, wy)

        while not (wfccore.done or wfccore.contradiction):
            before = sum(len(s) for row in wfccore.allowed for s in row)
            wfccore.step()
            after = sum(len(s) for row in wfccore.allowed for s in row)
            if after == before:
                break
        if wfccore.done:
            self.wfc = wfccore.allowed
