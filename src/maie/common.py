from enum import IntEnum
try:
    from typing import TypeAlias
except ImportError:
    from typing import Any
    TypeAlias = Any

Vec2: TypeAlias = tuple[float, float]
GVec2: TypeAlias = tuple[int, int]
Bounds: TypeAlias = tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
Segment: TypeAlias = tuple[Vec2, Vec2]
Size: TypeAlias = tuple[int, int]
Color: TypeAlias = tuple[int, int, int]


def clamp(val, a, b):
    return max(min(val, b), a)


class NeighborIndex(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
