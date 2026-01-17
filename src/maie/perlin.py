import numpy as np


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep01(t):
    return t * t * (3.0 - 2.0 * t)


def fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def hash32(ix, iy, seed=0):
    h = (np.uint64(ix) * np.uint64(374761393) +
         np.uint64(iy) * np.uint64(668265263) +
         np.uint64(seed) * np.uint64(974369323))

    h ^= (h >> np.uint64(13))
    h *= np.uint64(1274126177)
    h ^= (h >> np.uint64(16))
    return h


def gradient(ix, iy, seed=0):
    h = hash32(ix, iy, seed)
    # map [0, 2^32-1] -> [0, 2π)
    # angle = (h.astype(np.float64) / np.float64(0xFFFFFFFF)) * (2.0 * np.pi)
    angle = (h.astype(np.float64) * (1.0 / 2**32)) * (2.0 * np.pi)
    gx = np.cos(angle)
    gy = np.sin(angle)
    return gx, gy


def perlin2d(x, y, seed=0):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    fx = x - x0
    fy = y - y0

    g00x, g00y = gradient(x0, y0, seed)
    g10x, g10y = gradient(x1, y0, seed)
    g01x, g01y = gradient(x0, y1, seed)
    g11x, g11y = gradient(x1, y1, seed)

    dx00 = fx
    dy00 = fy
    dx10 = fx - 1.0
    dy10 = fy
    dx01 = fx
    dy01 = fy - 1.0
    dx11 = fx - 1.0
    dy11 = fy - 1.0

    n00 = g00x * dx00 + g00y * dy00
    n10 = g10x * dx10 + g10y * dy10
    n01 = g01x * dx01 + g01y * dy01
    n11 = g11x * dx11 + g11y * dy11

    sx = fade(fx)
    sy = fade(fy)

    ix0 = lerp(n00, n10, sx)
    ix1 = lerp(n01, n11, sx)
    value = lerp(ix0, ix1, sy)

    return value


def perlin2d_fbm(x, y, octaves=6, lacunarity=2.0, gain=0.5, base_freq=0.001, seed=0):
    amp = 1.0
    freq = base_freq
    value = 0.0
    for o in range(octaves):
        value += amp * perlin2d(x * freq, y * freq, seed + o)
        freq *= lacunarity
        amp *= gain
    return value
