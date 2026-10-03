"""noise-simplex, ported to Mojo.

`snoise2`, `snoise3` and `snoise4` keep upstream's names, argument order and
defaults, so this is a drop-in for the simplex half of the `noise` package.
Each accepts scalars exactly as upstream does, and additionally accepts any
broadcastable NumPy arrays, in which case it returns an array.

Upstream's entry points parse their arguments with `PyArg_ParseTuple` and a
`"fff"` format, so every coordinate it ever sees is a C `float` rounded from
the Python double. The kernels here are `float` too, and this layer does the
same rounding before handing anything over, which is what makes the results
agree with upstream bit for bit rather than to within a float32 epsilon.
"""

from __future__ import annotations

import numpy as np

from . import _lib

__all__ = ["snoise2", "snoise3", "snoise4", "FLT_MAX", "__version__"]

__version__ = "0.1.0"

# Upstream's argument parser defaults `repeatx`/`repeaty` to FLT_MAX and treats
# that as "no tiling". `None` is accepted as the same thing because upstream's
# docstrings spell the default that way.
FLT_MAX = float(np.finfo(np.float32).max)


def _check_octaves(octaves) -> int:
    n = int(octaves)
    if n <= 0:
        raise ValueError("Expected octaves value > 0")
    return n


def _tile(value) -> float:
    return FLT_MAX if value is None else value


def _scalar(a) -> bool:
    return np.ndim(a) == 0


def _shape(*coords) -> tuple:
    return np.broadcast_shapes(*[np.shape(c) for c in coords])


def _grid(shape, *values):
    """C-contiguous flat float32 copies of the broadcast coordinates.

    float32 because that is the type upstream's `PyArg_ParseTuple` format string
    narrows every coordinate to before the C ever sees it.
    """
    return [
        np.ascontiguousarray(
            np.broadcast_to(np.asarray(v, dtype=np.float64), shape),
            dtype=np.float32,
        ).reshape(-1)
        for v in values
    ]


def snoise2(x, y, octaves=1, persistence=0.5, lacunarity=2.0,
            repeatx=FLT_MAX, repeaty=FLT_MAX, base=0.0):
    """2D simplex noise value for the specified 2D coordinate.

    octaves -- specifies the number of passes, defaults to 1 (simple noise).

    persistence -- specifies the amplitude of each successive octave relative
    to the one below it. Defaults to 0.5 (each higher octave's amplitude
    is halved). Note the amplitude of the first pass is always 1.0.

    lacunarity -- specifies the frequency of each successive octave relative
    to the one below it, similar to persistence. Defaults to 2.0.

    repeatx, repeaty -- specifies the interval along each axis when
    the noise values repeat. This can be used as the tile size for creating
    tileable textures. `None` means no tiling, as does the default FLT_MAX.

    base -- specifies a fixed offset for the noise coordinates. Useful for
    generating different noise textures with the same repeat interval.
    """
    octaves = _check_octaves(octaves)
    repeatx = _tile(repeatx)
    repeaty = _tile(repeaty)
    # ctypes narrows each of these to c_float on the way in, which is the same
    # narrowing upstream's "fff|ifffff" format string performs.
    persistence = float(persistence)
    lacunarity = float(lacunarity)
    base = float(base)

    if _scalar(x) and _scalar(y):
        return _lib.lib().mns_noise2(float(x), float(y), octaves, persistence,
                                     lacunarity, repeatx, repeaty, base)

    shape = _shape(x, y)
    xs, ys = _grid(shape, x, y)
    out = np.empty(xs.size, dtype=np.float32)
    _lib.lib().mns_noise2_field(xs.ctypes.data, ys.ctypes.data, out.ctypes.data,
                                out.size, octaves, persistence, lacunarity,
                                repeatx, repeaty, base)
    return out.reshape(shape)


def snoise3(x, y, z, octaves=1, persistence=0.5, lacunarity=2.0):
    """3D simplex noise value for the specified 3D coordinate.

    octaves -- specifies the number of passes, defaults to 1 (simple noise).

    persistence -- specifies the amplitude of each successive octave relative
    to the one below it. Defaults to 0.5 (each higher octave's amplitude
    is halved). Note the amplitude of the first pass is always 1.0.

    lacunarity -- specifies the frequency of each successive octave relative
    to the one below it, similar to persistence. Defaults to 2.0.
    """
    octaves = _check_octaves(octaves)
    persistence = float(persistence)
    lacunarity = float(lacunarity)

    if _scalar(x) and _scalar(y) and _scalar(z):
        return _lib.lib().mns_noise3(float(x), float(y), float(z), octaves,
                                     persistence, lacunarity)

    shape = _shape(x, y, z)
    xs, ys, zs = _grid(shape, x, y, z)
    out = np.empty(xs.size, dtype=np.float32)
    _lib.lib().mns_noise3_field(xs.ctypes.data, ys.ctypes.data, zs.ctypes.data,
                                out.ctypes.data, out.size, octaves, persistence,
                                lacunarity)
    return out.reshape(shape)


def snoise4(x, y, z, w, octaves=1, persistence=0.5, lacunarity=2.0):
    """4D simplex noise value for the specified 4D coordinate.

    octaves -- specifies the number of passes, defaults to 1 (simple noise).

    persistence -- specifies the amplitude of each successive octave relative
    to the one below it. Defaults to 0.5 (each higher octave's amplitude
    is halved). Note the amplitude of the first pass is always 1.0.

    lacunarity -- specifies the frequency of each successive octave relative
    to the one below it, similar to persistence. Defaults to 2.0.
    """
    octaves = _check_octaves(octaves)
    persistence = float(persistence)
    lacunarity = float(lacunarity)

    if _scalar(x) and _scalar(y) and _scalar(z) and _scalar(w):
        return _lib.lib().mns_noise4(float(x), float(y), float(z), float(w),
                                     octaves, persistence, lacunarity)

    shape = _shape(x, y, z, w)
    xs, ys, zs, ws = _grid(shape, x, y, z, w)
    out = np.empty(xs.size, dtype=np.float32)
    _lib.lib().mns_noise4_field(xs.ctypes.data, ys.ctypes.data, zs.ctypes.data,
                                ws.ctypes.data, out.ctypes.data, out.size,
                                octaves, persistence, lacunarity)
    return out.reshape(shape)
