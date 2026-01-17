from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
import pygame
from typing import Iterable

from maie.camera import RenderContext, DrawLayer, draw_tile
from maie.common import Vec2, Color, GVec2, clamp
from maie.perlin import perlin2d, perlin2d_fbm
from maie.poisson import poisson_disc_2d
import math
import random

@dataclass
class MarsConfig:
    width: int = 1024
    height: int = 768
    tile_size: float = 4.0
    seed: int = 42
    
    # Layer specific configs
    min_crater_radius: float = 10.0
    max_crater_radius: float = 60.0 # Large craters
    crater_density_radius: float = 25.0 # For poisson sampling
    terrain_smoothing: float = 40.0 # Sigma for Voronoi blending (Higher = smoother, Lower = clearly defined plates)
    max_craters: int = 50 # Max number of craters to generate
    sea_level: float = 0.2 # Target level for initial fill (used in instant gen)
    num_rivers: int = 2000 # (Unused in Volumetric)
    river_sim_speed: int = 5 # (Unused in Volumetric)
    rain_rate: float = 0.1 # Water added per frame to sources
    sim_iterations: int = 20 # Physics steps per frame (Speed up)

class MarsLayer(Enum):
    POINTS = auto()
    REGIONS = auto()
    BASE = auto() # Raw Terrain
    CRATERS = auto() # With Craters
    WATER = auto() # Hydrology

class MarsWorld:
    def __init__(self, cfg: MarsConfig = MarsConfig()):
        self.cfg = cfg
        self.width = cfg.width
        self.height = cfg.height
        self.ts = cfg.tile_size
        
        # State
        self.enabled_layers: set[MarsLayer] = {l for l in MarsLayer} # Enable all by default
        
        self.points = []
        self.region_map = None 
        # Data arrays (initialized in _regenerate)
        self.points = []
        self.region_map = None 
        self.region_colors = {}
        self.region_heights = {}
        self.raw_terrain = None # Just Voronoi
        self.final_map = None # With Craters
        self.water_map = None # Lake Depth
        self.river_map = None # Flow Accumulation
        
        self._regenerate()

    @property
    def shape_tiles(self):
        return int(self.width / self.ts), int(self.height / self.ts)

    def _regenerate(self):
        print(f"Regenerating Mars World with seed {self.cfg.seed}...")
        shape = self.shape_tiles
        
        # Initialize empty arrays
        self.points = []
        self.region_map = np.zeros(shape, dtype=int)
        self.raw_terrain = np.zeros(shape)
        self.final_map = np.zeros(shape)
        self.water_map = np.zeros(shape)
        self.river_map = np.zeros(shape, dtype=int)
        
        # Simulation State
        self.sim_active = False
        self.sim_drops_done = 0
        self.sim_active = False
        self.sim_drops_done = 0
        self.sim_rng = random.Random(self.cfg.seed)
        self.rain_enabled = True # Runtime toggle
        
        # Pipeline
        self._generate_height() # generates raw_terrain
        self._generate_craters() # compositions to final_map
        self._generate_water() # hydrology analysis
        
    def reset_hydrology(self):
        print("Resetting Hydrology (Volumetric)...")
        shape = self.shape_tiles
        self.water_map = np.zeros(shape)
        self.rain_points = []
        self.sim_active = True
        self.sim_tick = 0
        

    def update(self):
        if not self.sim_active: return
        if MarsLayer.WATER not in self.enabled_layers: return
        
        self.rain_points = [] # Reset for this frame
        
        for _ in range(self.cfg.sim_iterations):
            self._sim_step()
            
    def _sim_step(self):
        self.sim_tick += 1
        w, h = self.shape_tiles
        
        # 1. Rain
        if self.rain_enabled:
            drops = 200 
            rx = np.random.randint(0, w, drops)
            ry = np.random.randint(0, h, drops)
            terrain_vals = self.final_map[rx, ry]
            valid_rain = terrain_vals > 0.4
            
            # Apply Rain
            v_rx = rx[valid_rain]
            v_ry = ry[valid_rain]
            self.water_map[v_rx, v_ry] += self.cfg.rain_rate
            
            # Store for Vis (Optimized: just store the arrays)
            self.rain_points.append((v_rx, v_ry))
        
        # 2. Flow
        H = self.final_map + self.water_map
        water = self.water_map
        
        # Calc Flow
        H_padded = np.pad(H, 1, mode='edge')
        N = H_padded[0:-2, 1:-1]
        S = H_padded[2:, 1:-1]
        W = H_padded[1:-1, 0:-2]
        E = H_padded[1:-1, 2:]
        
        dN = np.maximum(0, H - N)
        dS = np.maximum(0, H - S)
        dW = np.maximum(0, H - W)
        dE = np.maximum(0, H - E)
        
        dTotal = dN + dS + dW + dE
        dTotal_safe = dTotal.copy()
        dTotal_safe[dTotal_safe == 0] = 1.0
        
        flow_cap = water * 0.5 
        total_flow_wanted = dTotal * 0.5
        outflow_total = np.minimum(flow_cap, total_flow_wanted)
        scale_factor = outflow_total / dTotal_safe
        
        fN = dN * scale_factor
        fS = dS * scale_factor
        fW = dW * scale_factor
        fE = dE * scale_factor
        
        net_change = np.zeros_like(water)
        net_change -= outflow_total
        net_change += np.roll(fS, 1, axis=0) # Inflow from North
        net_change += np.roll(fN, -1, axis=0) # Inflow from South
        net_change += np.roll(fE, 1, axis=1) # Inflow from West
        net_change += np.roll(fW, -1, axis=1) # Inflow from East
        
        self.water_map += net_change
        self.water_map = np.maximum(0, self.water_map)
        
    def _generate_height(self):
        print("Generating structured terrain (Voronoi/Regional)...")
        w_tiles, h_tiles = self.shape_tiles

        # 1. Define Regions (Poisson Disc)
        # We use a larger radius to get big distinct chunks (Plateaus/Basins)
        radius = 150.0  
        points = poisson_disc_2d(
            (0, 0, self.width, self.height),
            radius=radius,
            n_points=100, # Max points
            seed=self.cfg.seed
        )
        
        # 2. Assign Elevation & Roughness to Regions
        # Each region gets a target height and a roughness factor.
        rng = random.Random(self.cfg.seed)
        region_heights = {}
        region_roughness = {}
        
        for i, p in enumerate(points):
            nx = p[0] / self.width
            ny = p[1] / self.height
            
            # Coarse noise to decide biome type
            coarse = perlin2d(nx*2, ny*2, seed=self.cfg.seed) # -1..1
            coarse = (coarse + 1) * 0.5 # 0..1
            
            if coarse > 0.55:
                # Highland / Crustal Block
                # High elevation, Rough terrain
                h = 0.6 + rng.uniform(0.0, 0.4)
                r = 0.4 + rng.uniform(0.0, 0.3)
            else:
                # Lowland / Basin
                # Low elevation, Smooth terrain
                h = 0.1 + rng.uniform(0.0, 0.3)
                r = 0.05 + rng.uniform(0.0, 0.15)
                
            # Store
            region_heights[i] = h
            region_roughness[i] = r
            
        self.points = points
        self.region_heights = region_heights
        
        # Generate random colors for regions for debug visualization
        self.region_colors = {
            i: (rng.randint(50, 200), rng.randint(50, 200), rng.randint(50, 200))
            for i in range(len(points))
        }

        # 3. Interpolate Regions (IDW) - Vectorized
        print("Interpolating regions (Vectorized IDW)...")
        
        # Grid coordinates
        tx = np.arange(w_tiles) * self.ts + self.ts/2
        ty = np.arange(h_tiles) * self.ts + self.ts/2
        grid_x, grid_y = np.meshgrid(tx, ty, indexing='ij')
        
        # Flatten
        flat_x = grid_x.ravel()[:, np.newaxis]
        flat_y = grid_y.ravel()[:, np.newaxis]
        
        # Points
        pts_arr = np.array(points)
        pts_x = pts_arr[:, 0][np.newaxis, :]
        pts_y = pts_arr[:, 1][np.newaxis, :]
        
        # Distance Matrix
        dist_sq = (flat_x - pts_x)**2 + (flat_y - pts_y)**2
        
        # Distance Matrix
        dist_sq = (flat_x - pts_x)**2 + (flat_y - pts_y)**2
        
        # Capture Voronoi Map (Nearest Neighbor) for Visuals
        # We still want the "blocky" debug layer to show the logical regions
        nearest_1 = np.argmin(dist_sq, axis=1) # (N_pixels,)
        self.region_map = nearest_1.reshape((w_tiles, h_tiles))
        
        # Smooth Interpolation: Use ALL points (Global Blending)
        # Using k=4 creates discontinuities where the 4th/5th neighbor swap.
        # Using all points guarantees smoothness.
        
        # IDW Weights
        # Sigma controls how "tight" the regions are.
        # Small sigma = flat plateaus with steep cliffs
        # Large sigma = rolling hills
        sigma = self.cfg.terrain_smoothing
        weights = np.exp(-dist_sq / (2 * sigma * sigma))
        
        # Normalize
        w_sum = weights.sum(axis=1, keepdims=True)
        w_sum[w_sum < 1e-10] = 1.0 
        weights /= w_sum
        
        # Map indices to values
        h_arr = np.array([region_heights[i] for i in range(len(points))])
        r_arr = np.array([region_roughness[i] for i in range(len(points))])
        
        # Weighted sums using matrix multiplication
        # weights: (N_px, N_pts)
        # values: (N_pts,) -> broadcast
        # dot product is better: (N_px, N_pts) @ (N_pts, 1) -> (N_px, 1)
        
        flat_base = weights @ h_arr
        flat_amp  = weights @ r_arr
        
        # Reshape
        base_h = flat_base.reshape((w_tiles, h_tiles))
        amp_h = flat_amp.reshape((w_tiles, h_tiles))

        # 4. Generate Noise Modulation (UUWorld style)
        x = np.linspace(0, self.width, w_tiles)
        y = np.linspace(0, self.height, h_tiles)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        
        # Macro Noise (Large scale)
        macro = perlin2d_fbm(xx, yy, octaves=4, base_freq=4.0/self.width, seed=self.cfg.seed + 99)
        # macro is approx -1..1
        
        # 5. Combine: Base + Amplitude * Noise
        self.base_height = base_h + amp_h * macro
        
        # 6. Add Detail (Micro texture)
        detail = perlin2d_fbm(xx, yy, octaves=4, base_freq=20.0/self.width, seed=self.cfg.seed + 101)
        # Mix: 90% Structure, 10% Detail
        self.raw_terrain = self.base_height * 0.9 + detail * 0.1 # Use self.base_height from step 5
        self.raw_terrain = np.clip(self.raw_terrain, 0.0, 1.0)
            
        print(f"Base Height generated. Shape: {self.raw_terrain.shape}")

    def _generate_craters(self):
        print("Generating craters on Voronoi centers...")
        # Start from raw terrain
        self.final_map = self.raw_terrain.copy()
        
        rng = random.Random(self.cfg.seed + 20)
        w_tiles, h_tiles = self.shape_tiles
        
        # Grid fields for noise
        # Precompute noise for roughness?
        # Actually doing per-crater polar noise is cheap enough for ~50 craters.
        
        # 1. Start by collecting candidates (Lowlands)
        candidates = []
        for i, center in enumerate(self.points):
            h_val = self.region_heights.get(i, 0.5)
            # Threshold: If it's a lowland (e.g. < 0.4 or 0.5), make it a crater.
            if h_val < 0.45:
                candidates.append(center)
                
        # 2. Shuffle and Limit
        rng.shuffle(candidates)
        candidates = candidates[:self.cfg.max_craters]
        
        # 3. Stamp
        for center in candidates:
            # Radius: roughly related to the region size or random?
            # User said "big spheres".
            # Let's make them fairly large (e.g. 25-60)
            r = rng.uniform(25.0, 55.0)
            
            # Depth:
            # User wants "higher elevation difference"
            depth_scale = 0.4  # Much deeper vs 0.15
            rim_scale = 0.15   # Higher rim vs 0.05
            
            cx, cy = center[0] / self.ts, center[1] / self.ts
            r_tiles = r / self.ts
            
            # Bounds
            pad = 10
            x0 = int(max(0, cx - r_tiles - pad))
            x1 = int(min(w_tiles, cx + r_tiles + pad))
            y0 = int(max(0, cy - r_tiles - pad))
            y1 = int(min(h_tiles, cy + r_tiles + pad))
            
            if x1 <= x0 or y1 <= y0: continue
            
            # Grid
            ix = np.arange(x0, x1)
            iy = np.arange(y0, y1)
            ixx, iyy = np.meshgrid(ix, iy, indexing='ij')
            
            # Polar coords relative to center
            dx = ixx - cx
            dy = iyy - cy
            dist = np.sqrt(dx*dx + dy*dy)
            angle = np.arctan2(dy, dx) # -pi to pi
            
            # 2. Rounder Shape (Reduced Polar Noise)
            # User wants "more rounded", so we reduce the amplitude of the noise
            freq = rng.uniform(3.0, 8.0)
            phase = rng.uniform(0, 100)
            
            # Reduced coefficients: 0.05 instead of 0.15
            noise_val = 0.05 * np.sin(angle * freq + phase) + \
                        0.03 * np.sin(angle * freq * 2.3 + phase)
                        
            # Distorted distance metric
            dist_distorted = dist / (1.0 + noise_val)
            
            # Normalized t
            t = dist_distorted / r_tiles
            
            # Masks
            mask_inner = t < 1.0
            mask_rim = (t >= 1.0) & (t < 1.6)
            
            # Apply
            if np.any(mask_inner):
                # Bowl
                shape = (1.0 - t[mask_inner]**2.0)
                self.final_map[x0:x1, y0:y1][mask_inner] -= shape * depth_scale
                
            if np.any(mask_rim):
                # Rim
                tt = (t[mask_rim] - 1.0) / 0.6 # 0..1
                # Sharper rim falloff
                rim_shape = (1.0 - tt) * np.exp(-3.0 * tt)
                self.final_map[x0:x1, y0:y1][mask_rim] += rim_shape * rim_scale

        # Clamp final
        self.final_map = np.clip(self.final_map, 0.0, 1.0)
        print(f"Rough Craters blended.")

    def _generate_water(self):
        print("Initializing Hydrology (Volumetric Mode)...")
        # Just init the arrays
        shape = self.shape_tiles
        self.water_map = np.zeros(shape)
        # Start sim
        self.reset_hydrology()
        
    def reset_hydrology(self):
        print("Resetting Hydrology (Volumetric)...")
        shape = self.shape_tiles
        self.water_map = np.zeros(shape)
        self.sim_active = True
        self.sim_tick = 0



    def get_layers(self, _) -> list[DrawLayer]:
        # Ignores the numeric list passed by default Playground
        # Uses internal enabled_layers set
        
        
        layers = []
        
        if MarsLayer.POINTS in self.enabled_layers:
            layers.append(DrawLayer(z=5, label="Points", draw=self._draw_points))
            
        if MarsLayer.REGIONS in self.enabled_layers:
            layers.append(DrawLayer(z=8, label="Regions", draw=self._draw_regions))
        
        if MarsLayer.BASE in self.enabled_layers:
            layers.append(DrawLayer(z=10, label="Terrain", draw=self._draw_base_terrain))
            
        if MarsLayer.CRATERS in self.enabled_layers:
            layers.append(DrawLayer(z=30, label="Craters", draw=self._draw_final_map))
        
        if MarsLayer.WATER in self.enabled_layers:
            layers.append(DrawLayer(z=40, label="Water", draw=self._draw_water))
            
        return layers

    def debug_layers(self) -> list[DrawLayer]:
        return [
            DrawLayer(z=5, label="Points", draw=self._draw_points),
            DrawLayer(z=10, label="Terrain", draw=self._draw_base_terrain),
            DrawLayer(z=30, label="Craters", draw=self._draw_final_map),
            DrawLayer(z=40, label="Water", draw=self._draw_water),
        ]

    def _draw_points(self, ctx: RenderContext):
        # Draw Poisson Centers
        for p in self.points:
            # p is world coordinate (x, y)
            # screen coord
            sc = ctx.camera.world_to_screen(p)
            pygame.draw.circle(ctx.screen, (255, 50, 50), sc, 3) # Little red dots

    def _get_visible_bounds(self, ctx: RenderContext):
        # Calculate visible world bounds in tiles
        # 1. Screen bounds
        sw, sh = ctx.screen.get_size()
        tl_world = ctx.camera.screen_to_world((0, 0))
        br_world = ctx.camera.screen_to_world((sw, sh))
        
        # 2. Convert to Tile indices
        ts = self.ts
        tx0 = int(math.floor(tl_world[0] / ts))
        ty0 = int(math.floor(tl_world[1] / ts))
        tx1 = int(math.ceil(br_world[0] / ts)) + 1
        ty1 = int(math.ceil(br_world[1] / ts)) + 1
        
        # 3. Clamp to Map Constraints
        w_tiles, h_tiles = self.shape_tiles
        
        tx0 = max(0, tx0)
        ty0 = max(0, ty0)
        tx1 = min(w_tiles, tx1)
        ty1 = min(h_tiles, ty1)
        
        return tx0, tx1, ty0, ty1

    def _blit_slice(self, ctx: RenderContext, surface: pygame.Surface, tx0, ty0):
        # Blit a surface representing tiles starting at tx0, ty0
        ts = self.ts
        
        # World position of the top-left of this slice
        world_x = tx0 * ts
        world_y = ty0 * ts
        
        # Determine world size of the slice
        # surface.get_width() is in tiles (pixels of the surface)
        # So in world units: width * ts
        dim_tiles_w = surface.get_width()
        dim_tiles_h = surface.get_height()
        
        if dim_tiles_w == 0 or dim_tiles_h == 0:
            return

        world_w = dim_tiles_w * ts
        world_h = dim_tiles_h * ts
        
        # Screen Rect
        # We assume axis aligned (no rotation)
        p0 = ctx.camera.world_to_screen((world_x, world_y))
        p1 = ctx.camera.world_to_screen((world_x + world_w, world_y + world_h))
        
        # Ceil proper scaling to avoid gaps?
        # int() truncates. If we want smooth tiling, we might need a +1 buffer or fblitting.
        # But pygame blits to ints.
        
        scr_w = p1[0] - p0[0]
        scr_h = p1[1] - p0[1]
        
        # Optimize: If scale is 1:1, don't scale
        # But our "surface pixel" is a "tile".
        # If zoom puts 1 tile = 1 screen pixel, then scale is 1.0.
        # But ts=4. zoom=1 -> 1 tile = 4 screen pixels.
        
        if scr_w > 0 and scr_h > 0:
            # Scale
            scaled = pygame.transform.scale(surface, (scr_w, scr_h))
            ctx.screen.blit(scaled, p0)

    def _draw_regions(self, ctx: RenderContext):
        # Vectorized Region Draw with Culling
        if self.region_map is None: return
        
        tx0, tx1, ty0, ty1 = self._get_visible_bounds(ctx)
        if tx1 <= tx0 or ty1 <= ty0: return
        
        # Slice
        sub_map = self.region_map[tx0:tx1, ty0:ty1]
        
        # Create palette (cached ideally, but cheap enough for 100 regions)
        # Find max region ID from keys locally or assume generous bounds
        max_id = 120 
        if self.region_colors:
             max_id = max(max_id, max(self.region_colors.keys()))
             
        palette = np.zeros((max_id + 1, 3), dtype=np.uint8)
        for rid, col in self.region_colors.items():
            if rid <= max_id:
                palette[rid] = col
                
        # Index directly (Safe if region_map has valid IDs)
        # region_map values come from poisson points len. usually 100.
        
        pixels = palette[sub_map] # (W_slice, H_slice, 3)
        
        surf = pygame.surfarray.make_surface(pixels)
        self._blit_slice(ctx, surf, tx0, ty0)

    def _draw_base_terrain(self, ctx: RenderContext):
        if self.raw_terrain is not None:
            self._draw_map_vectorized(ctx, self.raw_terrain)
            
    def _draw_final_map(self, ctx: RenderContext):
        if self.final_map is not None:
            self._draw_map_vectorized(ctx, self.final_map)

    def _draw_map_vectorized(self, ctx: RenderContext, data_map):
        # View Culling
        tx0, tx1, ty0, ty1 = self._get_visible_bounds(ctx)
        if tx1 <= tx0 or ty1 <= ty0: return
        
        vals = data_map[tx0:tx1, ty0:ty1]
        
        # Constants
        th_low = 0.35
        th_high = 0.85
        
        c_low = np.array([255, 245, 225])
        c_mid = np.array([200, 70, 20])
        c_high = np.array([10, 5, 5])
        
        out = np.zeros(vals.shape + (3,), dtype=np.uint8)
        
        mask_low = vals < th_low
        if np.any(mask_low):
            t = vals[mask_low] / th_low
            t = t[..., np.newaxis]
            cols = c_low * (1.0 - t) + c_mid * t
            out[mask_low] = cols.astype(np.uint8)
            
        mask_high = ~mask_low
        if np.any(mask_high):
            t = (vals[mask_high] - th_low) / (th_high - th_low)
            t = np.clip(t, 0.0, 1.0)[..., np.newaxis]
            cols = c_mid * (1.0 - t) + c_high * t
            out[mask_high] = cols.astype(np.uint8)
            
        surf = pygame.surfarray.make_surface(out)
        self._blit_slice(ctx, surf, tx0, ty0)

    def _draw_water(self, ctx: RenderContext):
        if self.water_map is None: return
        
        tx0, tx1, ty0, ty1 = self._get_visible_bounds(ctx)
        if tx1 <= tx0 or ty1 <= ty0: return
        
        # Helper Slice
        vals = self.water_map[tx0:tx1, ty0:ty1]
        
        # Check if empty (Optimization)
        mask_water = vals > 0.005
        has_water = np.any(mask_water)
        
        # If purely empty, skip
        if not has_water and not self.rain_points:
             return
             
        # Create pixels
        pixels = np.zeros(vals.shape + (3,), dtype=np.uint8)
        
        # Water Render
        if has_water:
            v = vals[mask_water]
            t = v / 0.15
            t = np.clip(t, 0.0, 1.0)[..., np.newaxis]
            
            c_shallow = np.array([40, 60, 150])
            c_deep = np.array([10, 10, 50])
            
            cols = c_shallow * (1.0 - t) + c_deep * t
            pixels[mask_water] = cols.astype(np.uint8)
            
        # Rain Render
        c_rain = np.array([180, 240, 255], dtype=np.uint8)
        w_slice, h_slice = pixels.shape[:2]
        
        for (rx, ry) in self.rain_points:
            # Global rain arrays. Filter to local slice.
            
            # Local coords
            lx = rx - tx0
            ly = ry - ty0
            
            # Valid mask
            valid = (lx >= 0) & (lx < w_slice) & (ly >= 0) & (ly < h_slice)
            
            if np.any(valid):
                pixels[lx[valid], ly[valid]] = c_rain
        
        surf = pygame.surfarray.make_surface(pixels)
        surf.set_colorkey((0, 0, 0))
        self._blit_slice(ctx, surf, tx0, ty0)




