import random
from collections import deque
from typing import Sequence, Protocol, Mapping
from typing import TypeAlias

import math
import numpy as np

from maie.common import NeighborIndex

TileIndex: TypeAlias = int
Allowed: TypeAlias = set[int]

CompatibilityIndex: TypeAlias = dict[TileIndex, dict[NeighborIndex, Allowed]]
NeighborDirection = {
    NeighborIndex.UP: (0, -1),
    NeighborIndex.RIGHT: (1, 0),
    NeighborIndex.DOWN: (0, 1),
    NeighborIndex.LEFT: (-1, 0),
}

Coords: TypeAlias = tuple[int, int]

EPS = 1e-12
MAX_ENTROPY = 1e9


def random_weighted_pick(rng, pool: Sequence[int], ws: Sequence[float]) -> int:
    r = rng.random() * sum(ws)
    acc = 0.0
    for i, w in zip(pool, ws):
        acc += w
        if r <= acc:
            choice = i
            break
    return choice


class TileSetLike(Protocol):
    def all_ids(self) -> Sequence:
        ...

    @property
    def weights(self) -> Mapping:
        ...

    def draw_tile(self, i, screen, x, y):
        ...

    def __len__(self):
        ...

    def is_allowed_neighbor(self, root, neighbor, direction) -> bool:
        ...


class WFCTile:
    def __init__(self, tileset: TileSetLike, width: int, height: int, weights=None, seed=None):
        self.ts = tileset
        self.w = width
        self.h = height
        self.rng = random.Random(seed)

        self.ids = list(self.ts.all_ids())
        self.weights = self._build_weight_dict(weights)

        self.entropies = np.zeros((height, width))
        self.allowed = np.full((height, width), set(self.ids), dtype=object)

        self._recompute_entropies_all()

        self.contradiction = False
        self.done = False

    def _build_weight_dict(self, weights):
        # base = from tileset, or uniform if not provided
        if hasattr(self.ts, "weights"):
            base = dict(self.ts.weights)
        else:
            base = {tid: 1.0 for tid in self.ids}

        # optional override:
        if weights is not None:
            if isinstance(weights, dict):
                base.update(weights)  # override per tile
            else:
                # sequence, same order as self.ids
                if len(weights) != len(self.ids):
                    raise ValueError("weights length mismatch")
                base = {tid: float(w) for tid, w in zip(self.ids, weights)}

        return base

    def _cell_entropy(self, s) -> float:
        if not s:
            return MAX_ENTROPY
        if len(s) == 1:
            return 0.0
        # Shannon entropy of weights
        tot = sum(self.weights[i] for i in s)
        p = [self.weights[i] / tot for i in s]
        return -sum(pi * math.log(pi + EPS) for pi in p)

    def _recompute_entropies_all(self):
        for y in range(self.h):
            for x in range(self.w):
                self.entropies[y][x] = self._cell_entropy(self.allowed[y][x])

    def _get_candidates(self) -> list[Coords] | None:
        """Find cells with minimum entropy."""
        minH = MAX_ENTROPY
        candidates = []
        for y in range(self.h):
            for x in range(self.w):
                allowed = self.allowed[y][x]

                if len(allowed) == 0:
                    self.contradiction = True
                    return

                if len(allowed) == 1:
                    continue

                H = self.entropies[y][x]
                if H < minH - 1e-12:
                    minH = H
                    candidates = [(x, y)]
                elif abs(H - minH) < EPS:
                    candidates.append((x, y))
        return candidates

    def observe(self) -> Coords | None:
        if self.done or self.contradiction:
            return

        candidates = self._get_candidates()
        if not candidates:
            self.done = True
            return

        x, y = self.rng.choice(candidates)
        allowed = self.allowed[y][x]

        pool = list(allowed)
        ws = [self.weights[t] for t in pool]
        choice = random_weighted_pick(self.rng, pool=pool, ws=ws)

        self.allowed[y][x] = {choice}
        self.entropies[y][x] = 0.0
        return x, y

    def propagate(self, start):
        if self.done or self.contradiction:
            return

        q = deque()
        q.append(start)

        while q:
            x, y = q.popleft()
            allowed_current = self.allowed[y][x]
            if not allowed_current:
                self.contradiction = True
                return

            for d, (dx, dy) in NeighborDirection.items():
                nx, ny = x + dx, y + dy

                # skip out-of-bounds neighbors
                if not (0 <= nx < self.w and 0 <= ny < self.h):
                    continue

                allowed_neighbor = self.allowed[ny][nx]

                # if no allowed tiles -> mark contradiction
                if not allowed_neighbor:
                    self.contradiction = True
                    return

                allow = self._allowed_for_neighbor(allowed_current, allowed_neighbor, d)

                if allow != allowed_neighbor:
                    self.allowed[ny][nx] = allow
                    self.entropies[ny][nx] = self._cell_entropy(allow)
                    q.append((nx, ny))

    def _allowed_for_neighbor(self, allowed_current: Allowed, allowed_neighbor: Allowed, d: NeighborIndex) -> Allowed:
        # neighbor must be compatible with at least one tile in allowed_xy
        allow = set()
        for n in allowed_neighbor:
            for c in allowed_current:
                if self.ts.is_allowed_neighbor(c, n, d):
                    allow.add(n)
                    break
        return allow

    def step(self):
        p = self.observe()
        if p is not None:
            self.propagate(p)

    def reset(self, seed):
        self.rng = random.Random(seed)
        self.allowed = np.full((self.h, self.w), set(self.ids), dtype=object)
        self._recompute_entropies_all()
        self.contradiction = False
        self.done = False
