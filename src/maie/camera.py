from dataclasses import dataclass
from typing import Callable

import pygame

from maie.common import Vec2, GVec2, Color


@dataclass(frozen=True)
class InputState:
    mouse_screen: Vec2
    mouse_world: Vec2


@dataclass
class RenderContext:
    screen: pygame.Surface
    camera: "Camera2D"
    input: InputState
    debug: bool


@dataclass(frozen=True)
class DrawLayer:
    z: int
    label: str
    draw: Callable[[RenderContext], None]


class Camera2D:
    def __init__(self, offset=pygame.Vector2(0, 0), zoom: float = 1.0):
        self.offset = offset
        self.zoom = zoom

    def world_to_screen(self, p: Vec2) -> GVec2:
        v = pygame.Vector2(p[0], p[1]) * self.zoom + self.offset
        return int(v.x), int(v.y)

    def screen_to_world(self, p: GVec2) -> Vec2:
        v = (pygame.Vector2(p[0], p[1]) - self.offset) / self.zoom
        return float(v.x), float(v.y)

    def zoom_at(self, screen_pos: GVec2, zoom_factor: float) -> None:
        """Zoom keeping the world point under cursor fixed."""
        before = pygame.Vector2(self.screen_to_world(screen_pos))
        self.zoom = max(0.05, min(50.0, self.zoom * zoom_factor))
        after = pygame.Vector2(self.screen_to_world(screen_pos))
        self.offset += (after - before) * self.zoom


def draw_tile(ctx: RenderContext, tile: GVec2, tile_size: float, color: Color) -> None:
    ts = tile_size
    x, y = tile
    p0 = ctx.camera.world_to_screen((x*ts, y*ts))
    p1 = ctx.camera.world_to_screen((x*ts+ts, y*ts+ts))
    w, h = p1[0]-p0[0], p1[1]-p0[1]
    rect = pygame.Rect(p0[0], p0[1], w, h)
    pygame.draw.rect(ctx.screen, color, rect)
