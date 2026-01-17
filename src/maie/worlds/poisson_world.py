import math
import numpy as np
import random

from typing import Iterable
import pygame
from dataclasses import dataclass

from maie.camera import RenderContext, DrawLayer
from maie.common import Vec2, Color, GVec2
from maie.poisson import poisson_disc_2d

VORONOI_CENTER_WIDTH = 5
VORONOI_CENTER_RADIUS = 5
VORONOI_CENTER_COLOR = (255, 255, 255)

COLOR_CENTER_HIGHLIGHT = (255, 255, 0)
COLOR_WORLD_EDGE = (150, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (00, 00, 255)


@dataclass
class PoissonWorldConfig:
    width: int = 640
    height: int = 480
    tile_size: float = 8.
    seed: int = 6
    poisson_n_points: int = 400
    poisson_radius: float = 30


class PoissonWorld:
    def __init__(self, cfg: PoissonWorldConfig):
        self.width = cfg.width
        self.height = cfg.height
        self.ts = cfg.tile_size
        self.cfg = cfg

        self._regenerate()

    def get_layers(self, which: Iterable[int]) -> list[DrawLayer]:
        return [
            DrawLayer(z=10, label="border", draw=self._draw_world_border),
        ]

    def debug_layers(self):
        return [
            DrawLayer(z=20, label="random", draw=self._draw_points),
            DrawLayer(z=30, label="poisson", draw=self._draw_poisson),
        ]

    def _draw_poisson(self, ctx: RenderContext):
        for p in self.region_centers.values():
            x = ctx.camera.world_to_screen(p)
            pygame.draw.circle(ctx.screen, VORONOI_CENTER_COLOR, x, VORONOI_CENTER_RADIUS, VORONOI_CENTER_WIDTH)

    def _draw_points(self, ctx: RenderContext):
        for p in self.points:
            x = ctx.camera.world_to_screen(p)
            pygame.draw.circle(ctx.screen, VORONOI_CENTER_COLOR, x, VORONOI_CENTER_RADIUS, VORONOI_CENTER_WIDTH)

    def _draw_world_border(self, ctx: RenderContext) -> None:
        w, h = self.width, self.height
        origin = ctx.camera.world_to_screen((0, 0))
        corner = ctx.camera.world_to_screen((w, h))
        pygame.draw.line(ctx.screen, COLOR_WORLD_EDGE, origin, (corner[0], origin[1]), 3)  # x-axis
        pygame.draw.line(ctx.screen, COLOR_WORLD_EDGE, origin, (origin[0], corner[1]), 3)  # x-axis
        pygame.draw.line(ctx.screen, COLOR_WORLD_EDGE, corner, (origin[0], corner[1]), 3)  # x-axis
        pygame.draw.line(ctx.screen, COLOR_WORLD_EDGE, corner, (corner[0], origin[1]), 3)  # x-axis

    def _site_color(self, i: int) -> Color:
        rng = random.Random(self.cfg.seed + i)
        return rng.randint(50, 205), rng.randint(50, 205), rng.randint(50, 205)

    @property
    def shape_tiles(self):
        return int(math.floor(self.width / self.ts)), int(math.floor(self.height / self.ts))

    def _regenerate(self):
        self._generate_poisson()
        self._generate_points()

    def _tile_at_world(self, p: Vec2) -> GVec2:
        return int(math.floor(p[0] / self.ts)), int(math.floor(p[1] / self.ts))

    def _generate_poisson(self):
        sites = poisson_disc_2d(
            (0, 0, self.cfg.width, self.cfg.height),
            radius=self.cfg.poisson_radius,
            n_points=self.cfg.poisson_n_points,
            seed=self.cfg.seed
        )

        centered = []
        for site in sites:
            tx, ty = self._tile_at_world(site)
            centered.append(((tx + 0.5) * self.ts, (ty + 0.5) * self.ts))

        self.region_centers = { i: site for i, site in enumerate(centered)}

        self.site_colors = {}
        for owner in self.region_centers:
            self.site_colors[owner] = self._site_color(owner)

    def _generate_points(self):
        rng = np.random.default_rng(self.cfg.seed)
        n = self.cfg.poisson_n_points
        self.points = np.column_stack((
            rng.uniform(0, self.width, n),
            rng.uniform(0, self.height, n)
        ))

