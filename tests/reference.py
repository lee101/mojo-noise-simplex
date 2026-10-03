"""NumPy transcription of upstream's `_simplex.c` + `_noise.h`, in float32.

This exists for two reasons:

  * the conda-forge `noise` 1.2.2 build segfaults in `snoise4` whenever
    `lacunarity` is supplied, so there is no upstream answer for that case;
  * it is a second, independent reference, so the port is not validated only
    against the binary it is meant to replace.

It follows the C source statement for statement, including the `fast_sin`
wrap trick and its float32 rounding.
"""

import numpy as np

F2 = np.float32(0.3660254037844386)
G2 = np.float32(0.21132486540518713)
F3 = np.float32(1.0 / 3.0)
G3 = np.float32(1.0 / 6.0)
F4 = np.float32(0.30901699437494745)
G4 = np.float32(0.1381966011250105)
M_1_PI = np.float32(1.0 / np.pi)
WRAP = np.float32(25165824.0)

GRAD3 = np.array([
    [1, 1, 0],
    [-1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0],
    [1, 0, 1],
    [-1, 0, 1],
    [1, 0, -1],
    [-1, 0, -1],
    [0, 1, 1],
    [0, -1, 1],
    [0, 1, -1],
    [0, -1, -1],
    [1, 0, -1],
    [-1, 0, -1],
    [0, -1, 1],
    [0, 1, 1],
], dtype=np.int32)

GRAD4 = np.array([
    [0, 1, 1, 1],
    [0, 1, 1, -1],
    [0, 1, -1, 1],
    [0, 1, -1, -1],
    [0, -1, 1, 1],
    [0, -1, 1, -1],
    [0, -1, -1, 1],
    [0, -1, -1, -1],
    [1, 0, 1, 1],
    [1, 0, 1, -1],
    [1, 0, -1, 1],
    [1, 0, -1, -1],
    [-1, 0, 1, 1],
    [-1, 0, 1, -1],
    [-1, 0, -1, 1],
    [-1, 0, -1, -1],
    [1, 1, 0, 1],
    [1, 1, 0, -1],
    [1, -1, 0, 1],
    [1, -1, 0, -1],
    [-1, 1, 0, 1],
    [-1, 1, 0, -1],
    [-1, -1, 0, 1],
    [-1, -1, 0, -1],
    [1, 1, 1, 0],
    [1, 1, -1, 0],
    [1, -1, 1, 0],
    [1, -1, -1, 0],
    [-1, 1, 1, 0],
    [-1, 1, -1, 0],
    [-1, -1, 1, 0],
    [-1, -1, -1, 0],
], dtype=np.int32)

SIMPLEX = np.array([
    [0, 1, 2, 3],
    [0, 1, 3, 2],
    [0, 0, 0, 0],
    [0, 2, 3, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1, 2, 3, 0],
    [0, 2, 1, 3],
    [0, 0, 0, 0],
    [0, 3, 1, 2],
    [0, 3, 2, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1, 3, 2, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1, 2, 0, 3],
    [0, 0, 0, 0],
    [1, 3, 0, 2],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [2, 3, 0, 1],
    [2, 3, 1, 0],
    [1, 0, 2, 3],
    [1, 0, 3, 2],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [2, 0, 3, 1],
    [0, 0, 0, 0],
    [2, 1, 3, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [2, 0, 1, 3],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [3, 0, 1, 2],
    [3, 0, 2, 1],
    [0, 0, 0, 0],
    [3, 1, 2, 0],
    [2, 1, 0, 3],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [3, 1, 0, 2],
    [0, 0, 0, 0],
    [3, 2, 0, 1],
    [3, 2, 1, 0],
], dtype=np.int32)

_PERM256 = np.array([
    [151, 160, 137, 91, 90, 15, 131, 13],
    [201, 95, 96, 53, 194, 233, 7, 225],
    [140, 36, 103, 30, 69, 142, 8, 99],
    [37, 240, 21, 10, 23, 190, 6, 148],
    [247, 120, 234, 75, 0, 26, 197, 62],
    [94, 252, 219, 203, 117, 35, 11, 32],
    [57, 177, 33, 88, 237, 149, 56, 87],
    [174, 20, 125, 136, 171, 168, 68, 175],
    [74, 165, 71, 134, 139, 48, 27, 166],
    [77, 146, 158, 231, 83, 111, 229, 122],
    [60, 211, 133, 230, 220, 105, 92, 41],
    [55, 46, 245, 40, 244, 102, 143, 54],
    [65, 25, 63, 161, 1, 216, 80, 73],
    [209, 76, 132, 187, 208, 89, 18, 169],
    [200, 196, 135, 130, 116, 188, 159, 86],
    [164, 100, 109, 198, 173, 186, 3, 64],
    [52, 217, 226, 250, 124, 123, 5, 202],
    [38, 147, 118, 126, 255, 82, 85, 212],
    [207, 206, 59, 227, 47, 16, 58, 17],
    [182, 189, 28, 42, 223, 183, 170, 213],
    [119, 248, 152, 2, 44, 154, 163, 70],
    [221, 153, 101, 155, 167, 43, 172, 9],
    [129, 22, 39, 253, 19, 98, 108, 110],
    [79, 113, 224, 232, 178, 185, 112, 104],
    [218, 246, 97, 228, 251, 34, 242, 193],
    [238, 210, 144, 12, 191, 179, 162, 241],
    [81, 51, 145, 235, 249, 14, 239, 107],
    [49, 192, 214, 31, 181, 199, 106, 157],
    [184, 84, 204, 176, 115, 121, 50, 45],
    [127, 4, 150, 254, 138, 236, 205, 93],
    [222, 114, 67, 29, 24, 72, 243, 141],
    [128, 195, 78, 66, 215, 61, 156, 180],
], dtype=np.int32)

_PERM256 = _PERM256.ravel()
PERM = np.concatenate([_PERM256, _PERM256])

f32 = np.float32


def fast_sin(x):
    # upstream: static inline float fast_sin(float x)
    x = f32(x)
    z = f32(x + WRAP)
    w = f32(x - f32(z - WRAP))
    y = f32(w - f32(w * np.abs(w)))
    return f32(y * f32(f32(3.1) + f32(f32(3.6) * np.abs(y))))


def fast_cos(x):
    return fast_sin(f32(x + f32(0.5)))


def noise2(x, y):
    x, y = f32(x), f32(y)
    s = f32((x + y) * F2)
    i = np.floor(f32(x + s))
    j = np.floor(f32(y + s))
    t = f32((i + j) * G2)

    xx = np.empty(3, dtype=np.float32)
    yy = np.empty(3, dtype=np.float32)
    f = np.empty(3, dtype=np.float32)
    noise = np.zeros(3, dtype=np.float32)

    xx[0] = f32(x - f32(i - t))
    yy[0] = f32(y - f32(j - t))
    i1 = int(xx[0] > yy[0])
    j1 = int(xx[0] <= yy[0])
    xx[2] = f32(f32(xx[0] + f32(G2 * 2.0)) - 1.0)
    yy[2] = f32(f32(yy[0] + f32(G2 * 2.0)) - 1.0)
    xx[1] = f32(f32(xx[0] - i1) + G2)
    yy[1] = f32(f32(yy[0] - j1) + G2)

    I = int(i) & 255
    J = int(j) & 255
    g = [PERM[I + PERM[J]] % 12,
         PERM[I + i1 + PERM[J + j1]] % 12,
         PERM[I + 1 + PERM[J + 1]] % 12]

    for c in range(3):
        f[c] = f32(f32(0.5) - f32(f32(xx[c] * xx[c]) + f32(yy[c] * yy[c])))
    for c in range(3):
        if f[c] > 0:
            noise[c] = f32(f32(f[c] ** 4) * f32(
                f32(GRAD3[g[c]][0] * xx[c]) + f32(GRAD3[g[c]][1] * yy[c])))
    return f32(f32(noise[0] + noise[1] + noise[2]) * 70.0)


def noise3(x, y, z):
    x, y, z = f32(x), f32(y), f32(z)
    s = f32(f32(f32(x + y) + z) * F3)
    i = np.floor(f32(x + s))
    j = np.floor(f32(y + s))
    k = np.floor(f32(z + s))
    t = f32(f32(f32(i + j) + k) * G3)

    pos = np.zeros((4, 3), dtype=np.float32)
    noise = np.zeros(4, dtype=np.float32)
    f = np.empty(4, dtype=np.float32)
    o1 = np.zeros(3, dtype=np.int64)
    o2 = np.zeros(3, dtype=np.int64)

    pos[0][0] = f32(x - f32(i - t))
    pos[0][1] = f32(y - f32(j - t))
    pos[0][2] = f32(z - f32(k - t))

    if pos[0][0] >= pos[0][1]:
        if pos[0][1] >= pos[0][2]:
            o1[:] = (1, 0, 0)
            o2[:] = (1, 1, 0)
        elif pos[0][0] >= pos[0][2]:
            o1[:] = (1, 0, 0)
            o2[:] = (1, 0, 1)
        else:
            o1[:] = (0, 0, 1)
            o2[:] = (1, 0, 1)
    else:
        if pos[0][1] < pos[0][2]:
            o1[:] = (0, 0, 1)
            o2[:] = (0, 1, 1)
        elif pos[0][0] < pos[0][2]:
            o1[:] = (1 - 1, 1, 0)
            o2[:] = (0, 1, 1)
        else:
            o1[:] = (0, 1, 0)
            o2[:] = (1, 1, 0)

    for c in range(3):
        pos[3][c] = f32(f32(pos[0][c] - 1.0) + f32(f32(3.0) * G3))
        pos[2][c] = f32(f32(pos[0][c] - o2[c]) + f32(f32(2.0) * G3))
        pos[1][c] = f32(f32(pos[0][c] - o1[c]) + G3)

    I, J, K = int(i) & 255, int(j) & 255, int(k) & 255
    g = [PERM[I + PERM[J + PERM[K]]] % 12,
         PERM[I + o1[0] + PERM[J + o1[1] + PERM[o1[2] + K]]] % 12,
         PERM[I + o2[0] + PERM[J + o2[1] + PERM[o2[2] + K]]] % 12,
         PERM[I + 1 + PERM[J + 1 + PERM[K + 1]]] % 12]

    for c in range(4):
        v = f32(0.6)
        for q in range(3):
            v = f32(v - f32(pos[c][q] * pos[c][q]))
        f[c] = v
    for c in range(4):
        if f[c] > 0:
            d = f32(GRAD3[g[c]][0] * pos[c][0])
            d = f32(d + f32(GRAD3[g[c]][1] * pos[c][1]))
            d = f32(d + f32(GRAD3[g[c]][2] * pos[c][2]))
            noise[c] = f32(f32(f[c] ** 4) * d)
    return f32(f32(noise[0] + noise[1] + noise[2] + noise[3]) * 32.0)


def noise4(x, y, z, w):
    x, y, z, w = f32(x), f32(y), f32(z), f32(w)
    noise = np.zeros(5, dtype=np.float32)
    s = f32(f32(f32(f32(x + y) + z) + w) * F4)
    i = np.floor(f32(x + s))
    j = np.floor(f32(y + s))
    k = np.floor(f32(z + s))
    l = np.floor(f32(w + s))
    t = f32(f32(f32(f32(i + j) + k) + l) * G4)

    p = np.empty((5, 4), dtype=np.float32)
    p[0][0] = f32(x - f32(i - t))
    p[0][1] = f32(y - f32(j - t))
    p[0][2] = f32(z - f32(k - t))
    p[0][3] = f32(w - f32(l - t))

    c = (int(p[0][0] > p[0][1]) * 32 + int(p[0][0] > p[0][2]) * 16
         + int(p[0][1] > p[0][2]) * 8 + int(p[0][0] > p[0][3]) * 4
         + int(p[0][1] > p[0][3]) * 2 + int(p[0][2] > p[0][3]))
    row = SIMPLEX[c]
    o = [np.array([int(row[d] >= 3) for d in range(4)], dtype=np.float32),
         np.array([int(row[d] >= 2) for d in range(4)], dtype=np.float32),
         np.array([int(row[d] >= 1) for d in range(4)], dtype=np.float32),
         np.zeros(4, dtype=np.float32)]

    for d in range(1, 4):
        for q in range(4):
            p[d][q] = f32(f32(p[0][q] - o[d - 1][q]) + f32(f32(d) * G4))
    for q in range(4):
        p[4][q] = f32(f32(p[0][q] - 1.0) + f32(f32(4.0) * G4))

    I, J, K, L = int(i) & 255, int(j) & 255, int(k) & 255, int(l) & 255
    g = [PERM[I + PERM[J + PERM[K + PERM[L]]]] & 0x1F,
         PERM[I + int(o[0][0]) + PERM[J + int(o[0][1])
                                  + PERM[K + int(o[0][2]) + PERM[L + int(o[0][3])]]]] & 0x1F,
         PERM[I + int(o[1][0]) + PERM[J + int(o[1][1])
                                  + PERM[K + int(o[1][2]) + PERM[L + int(o[1][3])]]]] & 0x1F,
         PERM[I + int(o[2][0]) + PERM[J + int(o[2][1])
                                  + PERM[K + int(o[2][2]) + PERM[L + int(o[2][3])]]]] & 0x1F,
         PERM[I + 1 + PERM[J + 1 + PERM[K + 1 + PERM[L + 1]]]] & 0x1F]

    for d in range(5):
        v = f32(0.6)
        for q in range(4):
            v = f32(v - f32(p[d][q] * p[d][q]))
        if v >= 0.0:
            v = f32(v * v)
            dot = f32(GRAD4[g[d]][0] * p[d][0])
            for q in (1, 2, 3):
                dot = f32(dot + f32(GRAD4[g[d]][q] * p[d][q]))
            noise[d] = f32(f32(v * v) * dot)
    return f32(f32(27.0) * f32(noise[0] + noise[1] + noise[2] + noise[3] + noise[4]))


def fbm(fn, coords, octaves, persistence, lacunarity):
    coords = [f32(c) for c in coords]
    freq = f32(1.0)
    amp = f32(1.0)
    mx = f32(1.0)
    total = f32(fn(*coords))
    for _ in range(1, octaves):
        freq = f32(freq * lacunarity)
        amp = f32(amp * persistence)
        mx = f32(mx + amp)
        total = f32(total + f32(fn(*[f32(c * freq) for c in coords]) * amp))
    return f32(total / mx)


def snoise2(x, y, octaves=1, persistence=0.5, lacunarity=2.0, repeatx=None,
            repeaty=None, base=0.0):
    if repeatx is None and repeaty is None:
        freq = f32(1.0)
        amp = f32(1.0)
        mx = f32(1.0)
        total = noise2(f32(x + base), f32(y + base))
        for _ in range(1, octaves):
            freq = f32(freq * lacunarity)
            amp = f32(amp * persistence)
            mx = f32(mx + amp)
            total = f32(total + f32(noise2(f32(f32(x * freq) + base),
                                           f32(f32(y * freq) + base)) * amp))
        return f32(total / mx)
    z = f32(base)
    w = f32(base)
    if repeaty is not None:
        yf = f32(f32(y * 2.0) / f32(repeaty))
        yr = f32(f32(f32(repeaty) * M_1_PI) * 0.5)
        y = f32(fast_sin(yf) * yr)
        w = f32(w + f32(fast_cos(yf) * yr))
        # Upstream passes `w` here, not `z`: only the y branch has run, so
        # `w` holds base + vyz*yr while `z` is still just base. A transcription
        # that passes `z` disagrees with upstream by ~1.0 on every point.
        if repeatx is None:
            return fbm(noise3, (f32(x), f32(y), f32(w)), octaves, persistence,
                       lacunarity)
    if repeatx is not None:
        xf = f32(f32(x * 2.0) / f32(repeatx))
        xr = f32(f32(f32(repeatx) * M_1_PI) * 0.5)
        x = f32(fast_sin(xf) * xr)
        z = f32(z + f32(fast_cos(xf) * xr))
        if repeaty is None:
            return fbm(noise3, (f32(x), f32(y), f32(z)), octaves, persistence,
                       lacunarity)
    return fbm(noise4, (f32(x), f32(y), f32(z), f32(w)), octaves, persistence,
               lacunarity)


def snoise3(x, y, z, octaves=1, persistence=0.5, lacunarity=2.0):
    if octaves == 1:
        return noise3(x, y, z)
    return fbm(noise3, (x, y, z), octaves, persistence, lacunarity)


def snoise4(x, y, z, w, octaves=1, persistence=0.5, lacunarity=2.0):
    if octaves == 1:
        return noise4(x, y, z, w)
    return fbm(noise4, (x, y, z, w), octaves, persistence, lacunarity)
