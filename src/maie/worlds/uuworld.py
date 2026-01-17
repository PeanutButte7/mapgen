import math
import random

from typing import Iterable
import numpy as np
import pygame
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum, auto

from maie.camera import RenderContext, DrawLayer, draw_tile
from maie.common import Vec2, Color, GVec2, clamp
from maie.nearest_neighbors import NearestNeighbors
from maie.perlin import perlin2d, perlin2d_fbm
from maie.poisson import poisson_disc_2d
from maie.voronoi import voronoi_partition_tiles
from maie.dijkstra import dijkstra_grid, reconstruct_path_grid
from maie.wavefunctioncollapse import WFCTile
from maie.tilesets.city import get_tileset

COLOR_ROAD = (0, 0, 0)

VORONOI_CENTER_WIDTH = 2
VORONOI_CENTER_RADIUS = 2
VORONOI_CENTER_COLOR = (255, 255, 255)

COLOR_CENTER_HIGHLIGHT = (255, 255, 0)
COLOR_WORLD_EDGE = (150, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (00, 00, 255)

class ViewLayers(IntEnum):
    BORDER = 0
    VORONOI_CENTERS = auto()
    VORONOI = auto()
    NEIGHBORS = auto()
    NEAREST_CENTERS = auto()
    K_NEAREST_NEIGHBORS = auto()
    LAND_WATER = auto()
    CITY_CENTERS = auto()
    ROAD = auto()
    CITY = auto()

@dataclass
class UUWorldConfig:
    width: int = 2048
    height: int = 1536
    tile_size: float = 16.
    seed: int = 6
    poisson_n_points: int = 40
    poisson_radius: float = 300
    land_threshold: float = 0.4
    land_base_range: tuple[float, float] = (0.45, 0.65)
    land_amp_range: tuple[float, float] = (0.15, 0.50)
    land_uplift_range: tuple[float, float] = (0.00, 0.20)
    sea_base_range: tuple[float, float] = (0.10, 0.35)
    sea_amp_range: tuple[float, float] = (0.05, 0.18)
    sea_uplift_range: tuple[float, float] = (0.00, 0.05)
    k_nearest: int = 4
    sigma: float = 1000.
    n_cities: int = 8
    city_size_range: tuple[int, int] = (10, 25)


def colormap(t, base_color: Color) -> Color:
    r, g, b = base_color
    t = clamp(t, 0, 1)
    return int(r*t), int(g*t), int(b*t)


class UUWorld:
    def __init__(self, cfg: UUWorldConfig = UUWorldConfig()):
        self.width = cfg.width
        self.height = cfg.height
        self.ts = cfg.tile_size
        self.region_centers = None
        self.region_tile_owners = None
        self.cfg = cfg

        self._regenerate()

        self.layers = {
            ViewLayers.BORDER: DrawLayer(z=1, label="world_border", draw=self._draw_world_border),
            ViewLayers.VORONOI: DrawLayer(z=10, label="voronoi_cells", draw=self._draw_regions),
            ViewLayers.LAND_WATER: DrawLayer(z=15, label="land_and_water", draw=self._draw_land_and_water),
            ViewLayers.VORONOI_CENTERS: DrawLayer(z=20, label="voronoi_centers", draw=self._draw_voronoi_centers),
            ViewLayers.NEIGHBORS: DrawLayer(z=30, label="region_neighbors", draw=self._draw_region_neighbors),
            ViewLayers.ROAD: DrawLayer(z=35, label="road", draw=self._draw_road),
            ViewLayers.CITY: DrawLayer(z=40, label="city", draw=self._draw_wfc),
            ViewLayers.CITY_CENTERS: DrawLayer(z=45, label="city_centers", draw=self._draw_cities),
            ViewLayers.NEAREST_CENTERS: DrawLayer(z=50, label="current_center", draw=self._draw_current_region_center),
            ViewLayers.K_NEAREST_NEIGHBORS: DrawLayer(z=55, label="nearest_neighbors", draw=self._draw_nearest_neighbors),
        }

    def get_layers(self, which: Iterable[ViewLayers]) -> list[DrawLayer]:
        return [
            self.layers[l] for l in which
        ]

    def debug_layers(self):
        return [
            DrawLayer(z=30, label="ridge", draw=lambda ctx: self._draw_array(ctx, self.ridge)),
            DrawLayer(z=40, label="elev", draw=lambda ctx: self._draw_array(ctx, self.elevation)),
            DrawLayer(z=50, label="macro", draw=lambda ctx: self._draw_array(ctx, self.macro_noise)),
            DrawLayer(z=50, label="base", draw=lambda ctx: self._draw_array(ctx, self.base_map)),
            DrawLayer(z=80, label="land", draw=self._draw_land),
            DrawLayer(z=90, label="distances", draw=lambda ctx: self._draw_array(ctx, self.distances.transpose())),
        ]

    def _draw_land(self, ctx: RenderContext) -> None:
        ts = self.cfg.tile_size
        for tile, land in np.ndenumerate(self.is_land):
            draw_tile(ctx, tile, ts, COLOR_GREEN if land else COLOR_BLUE)


    def _draw_array(self, ctx: RenderContext, arr: np.ndarray):
        ts = self.cfg.tile_size
        mx = np.nanmax(arr)
        mn = np.nanmin(arr)
        normalized = (arr - mn) / (mx - mn)
        for tile, val in np.ndenumerate(normalized):
            if np.isnan(val):
                draw_tile(ctx, tile, ts, (200, 0, 0))
            else:
                draw_tile(ctx, tile, ts, (int(255 * val), int(255 * val), int(255 * val)))

    def _draw_regions(self, ctx: RenderContext):
        ts = self.cfg.tile_size
        for (tx, ty), owner in self.region_tile_owners.items():
            color = self.site_colors[owner]
            draw_tile(ctx, (tx, ty), ts, color)

    def _draw_voronoi_centers(self, ctx: RenderContext):
        for p in self.region_centers.values():
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

    def _draw_region_neighbors(self, ctx: RenderContext) -> None:
        simplices = self.neighbors.sol.simplices
        points = self.neighbors.sol.points

        for simplex in simplices:
            for i in range(3):
                p0 = points[simplex[i]]
                p1 = points[simplex[(i+1) % 3]]
                pygame.draw.line(ctx.screen, (0, 0, 0), ctx.camera.world_to_screen(p0), ctx.camera.world_to_screen(p1), 3)
                pygame.draw.line(ctx.screen, (255, 255, 255), ctx.camera.world_to_screen(p0), ctx.camera.world_to_screen(p1), 1)

    def _draw_land_and_water(self, ctx: RenderContext) -> None:
        ts = self.cfg.tile_size
        for tile, land in np.ndenumerate(self.is_land):
            color = COLOR_GREEN if land else COLOR_BLUE
            e_raw = self.elevation[tile]
            draw_tile(ctx, tile, ts, colormap(e_raw, color))

    @property
    def shape_tiles(self):
        return int(math.floor(self.width / self.ts)), int(math.floor(self.height / self.ts))

    def is_tile_in_bounds(self, tile: GVec2) -> bool:
        tw, th = self.shape_tiles
        x, y = tile
        return 0 <= x < tw and 0 <= y < th

    def _draw_wfc(self, ctx: RenderContext) -> None:
        for k, wfc in self.wfc_cities.items():
            _, city, size = self.cities[k]
            city_tile = self._tile_at_world(city)
            top_left = (city_tile[0]-size//2, city_tile[1]-size//2)

            for tile, val in np.ndenumerate(wfc):
                ty, tx = tile
                tlx, tly = top_left
                pos = (tx+tlx, ty+tly)
                if not self.is_tile_in_bounds(pos) or not self.is_land[pos] or self.road[pos]:
                    continue

                tt = next(iter(val))
                pos = (pos[0] * self.ts, pos[1] * self.ts)
                pos = ctx.camera.world_to_screen(pos)
                self.tileset.draw_tile(tt, ctx.screen, *pos, ctx.camera.zoom)
                # self._draw_tile(surface, pos, (0, 0, 255))

    def _tile_at_world(self, p: Vec2) -> GVec2:
        return int(math.floor(p[0] / self.ts)), int(math.floor(p[1] / self.ts))

    def _tile_to_world(self, tile: GVec2) -> Vec2:
        return tile[0] * self.ts, tile[1] * self.ts

    def _draw_current_region_center(self, ctx: RenderContext) -> None:
        tile = self._tile_at_world(ctx.input.mouse_world)
        owner = self.region_tile_owners.get(tile, None)
        if owner is None:
            return

        region_center_tile = self._tile_at_world(self.region_centers[owner])
        draw_tile(ctx, region_center_tile, self.ts, COLOR_CENTER_HIGHLIGHT)

    def _draw_nearest_neighbors(self, ctx: RenderContext) -> None:
        tile = self._tile_at_world(ctx.input.mouse_world)
        owner = self.region_tile_owners.get(tile, None)
        if owner is None:
            return

        nn = self.neighbors.nearest_neighbors(ctx.input.mouse_world, self.cfg.k_nearest)
        for p in nn:
            neighbor_tile = self._tile_at_world(p)
            draw_tile(ctx, neighbor_tile, self.ts, COLOR_CENTER_HIGHLIGHT)

    def _draw_cities(self, ctx: RenderContext) -> None:
        i = 0
        increment = 255 // len(self.cities)
        for region, pos, kind in self.cities:
            tile = self._tile_at_world(pos)
            draw_tile(ctx, tile, self.ts, color=(0, i * increment, i * increment))
            i += 1

    def _draw_road(self, ctx: RenderContext) -> None:
        ts = self.cfg.tile_size
        for tile, road in np.ndenumerate(self.road):
            if road:
                draw_tile(ctx, tile, ts, COLOR_ROAD)

    def _regenerate(self):
        self._generate_points()
        self._generate_areas()
        self._generate_terrain()
        self._generate_land_sea_map()
        self._calculate_elev_and_coast_dist()
        self._generate_location()
        self._calculate_cost_map()
        self._generate_paths()
        self._generate_wfc()

    def _generate_points(self):
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
        self.neighbors = NearestNeighbors(centered)

        self.site_colors = {}
        for owner in self.region_centers:
            self.site_colors[owner] = self._site_color(owner)

    def _generate_areas(self):
        owner_map, g_s, _ = voronoi_partition_tiles(
            (0, 0, self.cfg.width, self.cfg.height),
            self.region_centers.values(),
            self.ts
        )
        self.region_tile_owners = owner_map
        self.region_tiles = defaultdict(list)
        for tile, owner in self.region_tile_owners.items():
            self.region_tiles[owner].append(tile)

    def _generate_terrain(self):
        rng = random.Random(self.cfg.seed)
        perlin_offset = rng.uniform(-0.5, 0.5)

        self.is_land_pref = {}
        self.base = {}
        self.amp = {}
        self.uplift = {}

        self.base_map = np.empty(self.shape_tiles, dtype=float)
        self.amp_map = np.empty(self.shape_tiles, dtype=float)
        self.uplift_map = np.empty(self.shape_tiles, dtype=float)
        self.macro_noise = np.empty(self.shape_tiles, dtype=float)
        self.ridge = np.empty(self.shape_tiles, dtype=float)
        self.elevation = np.empty(self.shape_tiles, dtype=float)
        self.is_land = np.empty(self.shape_tiles, dtype=bool)

        for owner, site in self.region_centers.items():
            x, y = site
            val = 0.5 * (perlin2d(x / 128, y / 128, seed=self.cfg.seed) +1)
            is_land = val > self.cfg.land_threshold
            self.is_land_pref[owner] = is_land

            if is_land:
                self.base[owner] = rng.uniform(*self.cfg.land_base_range)
                self.amp[owner] = rng.uniform(*self.cfg.land_amp_range)
                self.uplift[owner] = rng.uniform(*self.cfg.land_uplift_range)
            else:
                self.base[owner] = rng.uniform(*self.cfg.sea_base_range)
                self.amp[owner] = rng.uniform(*self.cfg.sea_amp_range)
                self.uplift[owner] = rng.uniform(*self.cfg.sea_uplift_range)

        def b_to_f(val: bool) -> float:
            return 1.0 if val else -1.0

        for tile in np.ndindex(self.shape_tiles):

            point = self._tile_to_world(tile)
            k = self.cfg.k_nearest
            nn = self.neighbors.nearest_neighbors(point, k)
            owners = [self.region_tile_owners[self._tile_at_world(p)] for p in nn]

            weights = []
            for p in nn:
                diff = np.linalg.norm(np.array(tile - p))
                w = np.exp(-diff**2 / (2 * self.cfg.sigma**2))
                weights.append(w)
            weights = np.array([w / sum(weights) for w in weights])

            base = np.array([self.base[o] for o in owners])
            self.base_map[tile] = np.dot(weights, base)
            amp = np.array([self.amp[o] for o in owners])
            self.amp_map[tile] = np.dot(weights, amp)
            uplift = np.array([self.uplift[o] for o in owners])
            self.uplift_map[tile] = np.dot(weights, uplift)

            land_pref = np.array([b_to_f(self.is_land_pref[o]) for o in owners])
            self.is_land[tile] = np.dot(weights, land_pref) > 0.444

            x, y = point
            x += perlin_offset
            y += perlin_offset
            u = perlin2d_fbm(x, y, octaves=2, lacunarity=2, gain=0.5, seed=self.cfg.seed)
            self.ridge[tile] = (1 - np.abs(u))**2.5
            n0 = 0.5 * (perlin2d_fbm(x / 512, y / 512, octaves=3, lacunarity=2, gain=0.5, seed=self.cfg.seed) + 1)
            n1 = 0.5 * (perlin2d_fbm(x / 256, y / 256, octaves=2, seed=self.cfg.seed) + 1)
            n2 = 0.5 * (perlin2d_fbm(x / 64, y / 64, octaves=5, seed=self.cfg.seed) + 1)

            self.macro_noise[tile] = 0.5 * n0 + 0.3 * n1 + 0.2 * n2
            self.elevation[tile] = self.base_map[tile] + self.amp_map[tile] * (self.macro_noise[tile] - 0.5) + self.uplift_map[tile] * self.ridge[tile]

    def _generate_land_sea_map(self):
        shape = self.shape_tiles
        self.is_land = np.empty(shape, dtype=bool)
        for tile in np.ndindex(shape):
            owner = self.region_tile_owners[tile]
            self.is_land[tile] = self.is_land_pref[owner]

    def _calculate_elev_and_coast_dist(self):
        def avg(col):
            return sum(col) / len(col)

        def dist_sq(p1: Vec2 | GVec2, p2: Vec2 | GVec2):
            return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))

        self.mean_elevations = {}
        self.coast_dist = {}
        for owner, center in self.region_centers.items():
            self.mean_elevations[owner] = avg([self.elevation[tile] for tile in self.region_tiles[owner]])

            d = float("inf")
            if not self.is_land_pref[owner]:
                self.coast_dist[owner] = None
                continue

            for tile, is_land in np.ndenumerate(self.is_land):
                if is_land:
                    continue

                tile_center = ((tile[0] + 0.5) * self.ts, (tile[1] + 0.5) * self.ts )
                new_d = dist_sq(center, tile_center)
                if new_d < d:
                    d = new_d
                    self.coast_dist[owner] = (d, tile)

    def _is_eligible_for_city(self, region_owner: int) -> bool:
        if not self.is_land_pref[region_owner]:
            return False
        return self.mean_elevations[region_owner] < 0.9 and self.coast_dist[region_owner][0] < 250

    def _generate_location(self):
        rng = random.Random(self.cfg.seed)
        candidate_regions = [
            owner
            for owner in self.region_centers
            if self._is_eligible_for_city(owner)
        ]
        candidate_regions.sort(key=lambda o: self.coast_dist[o][0] + self.mean_elevations[o])
        city_regions = candidate_regions[:self.cfg.n_cities]

        self.cities = [(owner, self.region_centers[owner], rng.randint(*self.cfg.city_size_range)) for owner in city_regions]

    def _calculate_cost_map(self):
        shape = self.shape_tiles
        self.cost_map = np.ones(shape, dtype=float)
        for tile in np.ndindex(shape):
            if not self.is_land[tile]:
                self.cost_map[tile] = float("inf")
            else:
                if self.elevation[tile] > 0.8:
                    self.cost_map[tile] = 3.0
                if self.elevation[tile] > 0.5:
                    self.cost_map[tile] = 2.

    def _generate_paths(self):
        rng = random.Random(self.cfg.seed)

        def passable(i: int, j: int) -> bool:
            return self.is_land[i, j]

        def cost_of(i, j, nx, ny) -> float:
            return self.cost_map[(nx, ny)]

        cap = rng.randint(0, len(self.cities) - 1)
        _, capital, _ = self.cities[cap]

        capital = self._tile_at_world(capital)

        shape = self.shape_tiles
        distances, parent = dijkstra_grid(shape[0], shape[1], passable, cost_of, start=capital)
        self.road = np.zeros((shape), dtype=bool)
        self.distances = distances
        self.distances[np.isinf(self.distances)] = np.nan
        for i, (_, target, _) in enumerate(self.cities):
            if i == cap: continue
            target = self._tile_at_world(target)
            path = reconstruct_path_grid(parent, start=capital, goal=target)
            for tile in path:
                self.road[tile] = True

    def _generate_wfc(self):
        self.wfc_cities = {}
        for k, (_, city, size) in enumerate(self.cities):
            self.tileset = get_tileset(size=int(self.ts))
            wfccore = WFCTile(self.tileset, size, size)

            while not (wfccore.done or wfccore.contradiction):
                before = sum(len(s) for row in wfccore.allowed for s in row)
                wfccore.step()
                after = sum(len(s) for row in wfccore.allowed for s in row)
                if after == before:
                    break
            if wfccore.done:
                self.wfc_cities[k] =  wfccore.allowed
