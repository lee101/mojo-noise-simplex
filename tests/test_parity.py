"""Parity with upstream `noise` on the same inputs.

`noise` is the real upstream package (conda-forge `noise`, the `noise-simplex`
half of Casey Duncan's `noise`): it exposes `snoise2`/`snoise3`/`snoise4` as C
extensions over `_simplex.c` and `_noise.h`. Every numeric assertion here
compares this port against it, or against `tests/reference.py` (a NumPy
transcription of the same C source) in the two places upstream cannot be called
at all.

The tolerance is tight because the port is as close to upstream as float32
allows. Both compute in C `float`, the Python layer narrows inputs to float32
exactly as upstream's `PyArg_ParseTuple` format string does, and the skew
constants are upstream's decimal literals rather than recomputed ones. What is
left is float32 arithmetic noise (max observed 8.3e-7 over the sample sizes
below) plus the rare point that lands on the far side of a corner-selection
comparison such as `x0 > y0`, which is a different point in a different simplex
cell and so a genuinely different value. Those are counted, not averaged away:
`assert_parity` requires that almost every point agrees tightly and that the
disagreeing remainder stays under a stated budget.

For contrast, a version of this port that computed in float64 — more accurate
in the abstract — disagrees with upstream on 34% of points at |x| < 500 and 63%
at |x| < 5000, because the float64 lattice offset falls on the other side of
those corner tests.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

noise = pytest.importorskip("noise")
import reference as ref  # noqa: E402

import mojo_noise_simplex as mns  # noqa: E402

FM = mns.FLT_MAX

# Tight: this is the same arithmetic in the same width as upstream.
ATOL = 1e-5
# Points on the wrong side of a corner comparison. Measured rates are far below
# this even for 6-octave fBm at a lacunarity that pushes coordinates to ~3e3.
CORNER_BUDGET = 5e-3
# Aggressive lacunarity: coordinates reach ~3e3, where the float32 ULP of the
# skew offset is ~2e-4 and corner flips get noticeably more likely.
WILD_BUDGET = 5e-2


def coords(n, seed, lo=-20.0, hi=20.0):
    rng = np.random.default_rng(seed)
    return [rng.uniform(lo, hi, n).astype(np.float32) for _ in range(4)]


def _loop(fn, flat, count):
    return np.fromiter((fn(*row) for row in zip(*flat)), dtype=np.float64, count=count)


def _dispatch(raw, *args):
    if all(np.ndim(a) == 0 for a in args):
        return raw(*[float(a) for a in args])
    shape = np.broadcast_shapes(*[np.shape(a) for a in args])
    flat = [np.ravel(a) for a in np.broadcast_arrays(*args)]
    return _loop(raw, flat, flat[0].size).reshape(shape)


def up2(x, y, **kw):
    return _dispatch(
        lambda a, b: noise.snoise2(a, b, kw.get("octaves", 1),
                                   kw.get("persistence", 0.5),
                                   kw.get("lacunarity", 2.0),
                                   kw.get("repeatx", FM),
                                   kw.get("repeaty", FM),
                                   kw.get("base", 0.0)),
        x, y)


def up3(x, y, z, **kw):
    return _dispatch(
        lambda a, b, c: noise.snoise3(a, b, c, kw.get("octaves", 1),
                                      kw.get("persistence", 0.5),
                                      kw.get("lacunarity", 2.0)),
        x, y, z)


def up4(x, y, z, w, **kw):
    # Upstream's snoise4 segfaults on this build whenever `lacunarity` is passed
    # at all (7+ arguments), so it is never passed here either. The lacunarity
    # cases are checked against tests/reference.py instead.
    if kw.get("lacunarity", 2.0) != 2.0:
        raise ValueError("upstream snoise4 cannot be called with lacunarity")
    return _dispatch(
        lambda a, b, c, d: noise.snoise4(a, b, c, d, kw.get("octaves", 1),
                                         kw.get("persistence", 0.5)),
        x, y, z, w)


def assert_parity(ours, theirs, budget=CORNER_BUDGET):
    ours = np.asarray(ours, dtype=np.float64)
    theirs = np.asarray(theirs, dtype=np.float64)
    assert ours.shape == theirs.shape
    if ours.size == 0:
        return
    diff = np.abs(ours - theirs)
    disagree = diff > ATOL
    rate = disagree.mean()
    assert rate <= budget, (
        f"{rate:.4%} of points disagree by more than {ATOL} "
        f"(budget {budget:.4%}); worst {diff.max():.3e}")
    # Of the points that do agree, they must agree tightly, not marginally.
    if (~disagree).any():
        worst = diff[~disagree].max()
        assert worst <= ATOL, worst


# ------------------------------------------------------------------- scalars

def test_snoise2_scalar_matches_upstream():
    for x, y in [(0.0, 0.0), (0.5, 0.5), (0.3, 0.7), (-1.25, 4.5), (17.0, -9.0),
                 (1e4, -1e4), (0.123456789, 0.987654321)]:
        assert mns.snoise2(x, y) == pytest.approx(noise.snoise2(x, y), abs=ATOL)


def test_snoise3_scalar_matches_upstream():
    for x, y, z in [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.3, 0.7, 1.1),
                    (-1.25, 4.5, -6.75), (17.0, -9.0, 3.5)]:
        assert mns.snoise3(x, y, z) == pytest.approx(noise.snoise3(x, y, z), abs=ATOL)


def test_snoise4_scalar_matches_upstream():
    for x, y, z, w in [(0.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5, 0.5),
                       (0.3, 0.7, 1.1, 2.2), (-1.25, 4.5, -6.75, 0.125)]:
        assert mns.snoise4(x, y, z, w) == pytest.approx(noise.snoise4(x, y, z, w),
                                                        abs=ATOL)


@pytest.mark.parametrize("octaves", [1, 2, 3, 6, 12])
def test_octaves_match_upstream(octaves):
    xs, ys, zs, ws = coords(20_000, 5)
    assert_parity(mns.snoise2(xs, ys, octaves=octaves), up2(xs, ys, octaves=octaves))
    assert_parity(mns.snoise3(xs, ys, zs, octaves=octaves), up3(xs, ys, zs, octaves=octaves))
    assert_parity(mns.snoise4(xs, ys, zs, ws, octaves=octaves),
                  up4(xs, ys, zs, ws, octaves=octaves))


@pytest.mark.parametrize("persistence,lacunarity", [(0.1, 2.0), (0.9, 1.7),
                                                    (0.5, 3.5)])
def test_persistence_and_lacunarity_match_upstream(persistence, lacunarity):
    xs, ys, zs, _ = coords(20_000, 6)
    budget = WILD_BUDGET if lacunarity > 2.0 else CORNER_BUDGET
    assert_parity(
        mns.snoise3(xs, ys, zs, octaves=5, persistence=persistence, lacunarity=lacunarity),
        up3(xs, ys, zs, octaves=5, persistence=persistence, lacunarity=lacunarity),
        budget=budget)


@pytest.mark.parametrize("lo,hi", [(-20.0, 20.0), (-500.0, 500.0), (-5000.0, 5000.0)])
def test_parity_holds_at_large_coordinates(lo, hi):
    # The float32 ULP of the skew offset grows with the coordinate, so this is
    # the regime that separates a correctly-typed port from an accidentally
    # more accurate one.
    xs, ys, zs, ws = coords(10_000, 31, lo, hi)
    assert_parity(mns.snoise2(xs, ys), up2(xs, ys))
    assert_parity(mns.snoise3(xs, ys, zs), up3(xs, ys, zs))
    assert_parity(mns.snoise4(xs, ys, zs, ws), up4(xs, ys, zs, ws))
    assert_parity(mns.snoise3(xs, ys, zs, octaves=6),
                  up3(xs, ys, zs, octaves=6))


# --------------------------------------------------------------------- tiled

@pytest.mark.parametrize("repeatx,repeaty", [(8.0, FM), (FM, 8.0), (8.0, 8.0),
                                             (4.0, 16.0), (25.0, 8.0)])
@pytest.mark.parametrize("octaves", [1, 4])
def test_tiled_matches_upstream(repeatx, repeaty, octaves):
    xs, ys = coords(5_000, 7)[:2]
    assert_parity(
        mns.snoise2(xs, ys, octaves=octaves, repeatx=repeatx, repeaty=repeaty),
        up2(xs, ys, octaves=octaves, repeatx=repeatx, repeaty=repeaty))


def test_tiled_with_base_offset_matches_upstream():
    xs, ys = coords(5_000, 8)[:2]
    assert_parity(
        mns.snoise2(xs, ys, octaves=3, repeatx=6.0, repeaty=10.0, base=2.5),
        up2(xs, ys, octaves=3, repeatx=6.0, repeaty=10.0, base=2.5))


def test_tiled_periodicity():
    # A tiled field must repeat at the tile interval, which is the whole point
    # of repeatx/repeaty. Upstream's fast_sin is only piecewise smooth, so
    # compare this port against itself.
    xs = np.linspace(0.0, 8.0, 257)
    ys = np.linspace(0.0, 6.0, 257)
    a = mns.snoise2(xs, ys, repeatx=8.0, repeaty=6.0)
    b = mns.snoise2(xs + 8.0, ys + 6.0, repeatx=8.0, repeaty=6.0)
    assert np.array_equal(a, b)


def test_none_repeat_means_no_tiling():
    xs, ys = coords(50, 9)[:2]
    assert np.array_equal(mns.snoise2(xs, ys, repeatx=None, repeaty=None),
                          mns.snoise2(xs, ys))


# -------------------------------------------------------------------- fields

def test_field_matches_upstream_2d():
    xs, ys = coords(100_000, 10)[:2]
    assert_parity(mns.snoise2(xs, ys), up2(xs, ys))


def test_field_matches_upstream_3d():
    xs, ys, zs, _ = coords(100_000, 11)
    assert_parity(mns.snoise3(xs, ys, zs), up3(xs, ys, zs))


def test_field_matches_upstream_4d():
    xs, ys, zs, ws = coords(100_000, 12)
    assert_parity(mns.snoise4(xs, ys, zs, ws), up4(xs, ys, zs, ws))


def test_field_matches_reference_4d_lacunarity():
    # The case upstream cannot be asked about at all: its snoise4 crashes if
    # lacunarity is passed.
    xs, ys, zs, ws = coords(20_000, 13)
    ours = mns.snoise4(xs, ys, zs, ws, octaves=4, lacunarity=1.7)
    theirs = _loop(lambda a, b, c, d: float(ref.snoise4(a, b, c, d, 4, 0.5, 1.7)),
                   [xs, ys, zs, ws], xs.size)
    assert_parity(ours, theirs, budget=WILD_BUDGET)


def test_field_matches_reference_3d_aggressive_lacunarity():
    xs, ys, zs, _ = coords(20_000, 14)
    ours = mns.snoise3(xs, ys, zs, octaves=5, lacunarity=3.5)
    theirs = _loop(lambda a, b, c: float(ref.snoise3(a, b, c, 5, 0.5, 3.5)),
                   [xs, ys, zs], xs.size)
    assert_parity(ours, theirs, budget=WILD_BUDGET)


def test_serial_and_threaded_field_paths_agree():
    # PARALLEL_MIN is 262144 points, so this crosses the threaded driver.
    xs, ys, zs, _ = coords(300_000, 15)
    big = mns.snoise3(xs, ys, zs)
    small = mns.snoise3(xs[:1_000], ys[:1_000], zs[:1_000])
    assert np.array_equal(big[:1_000], small)
    assert_parity(big, up3(xs, ys, zs))


@pytest.mark.parametrize("n", [262_143, 262_144, 262_145, 262_147])
def test_threaded_threshold_is_bit_identical(n):
    # One point either side of PARALLEL_MIN, and one that leaves a ragged last
    # 16384-point chunk: the serial and threaded drivers must agree exactly.
    xs, ys, zs, _ = coords(n, 21)
    ours = mns.snoise3(xs, ys, zs)
    assert ours.size == n
    head = mns.snoise3(xs[:1_024], ys[:1_024], zs[:1_024])
    assert np.array_equal(ours[:1_024], head)
    assert_parity(ours, up3(xs, ys, zs))
    tiled = mns.snoise2(xs, ys, octaves=4, repeatx=8.0, repeaty=8.0)
    head_tiled = mns.snoise2(xs[:1_024], ys[:1_024], octaves=4,
                             repeatx=8.0, repeaty=8.0)
    assert np.array_equal(tiled[:1_024], head_tiled)
    assert_parity(tiled, up2(xs, ys, octaves=4, repeatx=8.0, repeaty=8.0))


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 17, 1_000])
def test_short_fields_cover_every_chunk_remainder(n):
    # Every worker takes a fixed 16384-point chunk, so short and ragged fields
    # are the case where an off-by-one in the chunk bounds would show up.
    xs, ys, zs, _ = coords(max(n, 1), 22)
    ours = mns.snoise3(xs[:n], ys[:n], zs[:n])
    assert ours.size == n
    if n:
        assert_parity(ours, up3(xs[:n], ys[:n], zs[:n]))


def test_field_matches_scalar_call():
    xs, ys, zs, _ = coords(1_000, 23)
    assert np.array_equal(mns.snoise2(xs, ys),
                          np.array([mns.snoise2(x, y) for x, y in zip(xs, ys)]))
    assert np.array_equal(mns.snoise3(xs, ys, zs),
                          np.array([mns.snoise3(x, y, z)
                                    for x, y, z in zip(xs, ys, zs)]))
    assert np.array_equal(mns.snoise2(xs, ys, octaves=3, repeatx=4.0, repeaty=4.0),
                          np.array([mns.snoise2(x, y, 3, 0.5, 2.0, 4.0, 4.0, 0.0)
                                    for x, y in zip(xs, ys)]))


def test_reference_agrees_with_upstream():
    """The NumPy transcription is only a useful second reference if it tracks
    the real thing, so check it against upstream on every entry point."""
    xs, ys, zs, ws = coords(2_000, 17)
    assert_parity(_loop(ref.noise2, [xs, ys], xs.size), up2(xs, ys))
    assert_parity(_loop(ref.noise3, [xs, ys, zs], xs.size), up3(xs, ys, zs))
    assert_parity(_loop(ref.noise4, [xs, ys, zs, ws], xs.size), up4(xs, ys, zs, ws))
    assert_parity(
        _loop(lambda a, b: float(ref.snoise2(a, b, 3, 0.5, 2.0, 4.0, 6.0, 0.0)),
              [xs, ys], xs.size),
        up2(xs, ys, octaves=3, repeatx=4.0, repeaty=6.0))
    # All three tiling branches, including the y-only one that passes `w`.
    assert_parity(
        _loop(lambda a, b: float(ref.snoise2(a, b, 3, 0.5, 2.0, None, 6.0, 0.0)),
              [xs, ys], xs.size),
        up2(xs, ys, octaves=3, repeatx=FM, repeaty=6.0))


def test_reference_rejects_none_as_no_tiling():
    # reference.py speaks upstream's docstring dialect (None = no tiling); the
    # package speaks the C one (FLT_MAX = no tiling). They must agree.
    xs, ys = coords(200, 24)[:2]
    a = np.array([float(ref.snoise2(x, y, 3, 0.5, 2.0, None, None, 0.0))
                  for x, y in zip(xs, ys)])
    b = mns.snoise2(xs, ys, octaves=3)
    assert_parity(a, b)


# -------------------------------------------------------------- API contract

def test_broadcasting():
    xs = np.linspace(0.0, 4.0, 64).reshape(8, 8)
    out = mns.snoise2(xs, 0.5)
    assert out.shape == (8, 8)
    assert_parity(out, up2(xs, 0.5))

    grid_x, grid_y = np.meshgrid(np.linspace(0, 3, 8), np.linspace(0, 3, 8))
    assert mns.snoise2(grid_x, grid_y).shape == (8, 8)


def test_mixed_scalar_and_array():
    xs = np.linspace(0.0, 2.0, 32) + 0.00137
    zs = np.linspace(-1.0, 1.0, 32) - 0.00091
    out = mns.snoise3(xs, 0.25, zs)
    assert out.shape == (32,)
    assert_parity(out, up3(xs, 0.25, zs))


def test_empty_input():
    empty = np.empty(0)
    assert mns.snoise2(empty, empty).shape == (0,)
    assert mns.snoise3(empty, empty, empty).shape == (0,)
    assert mns.snoise4(empty, empty, empty, empty).shape == (0,)


@pytest.mark.parametrize("octaves", [0, -1, -5])
def test_non_positive_octaves_raises_like_upstream(octaves):
    for call_, reference_call in (
            (lambda: mns.snoise2(0.1, 0.2, octaves=octaves),
             lambda: noise.snoise2(0.1, 0.2, octaves=octaves)),
            (lambda: mns.snoise3(0.1, 0.2, 0.3, octaves=octaves),
             lambda: noise.snoise3(0.1, 0.2, 0.3, octaves=octaves)),
            (lambda: mns.snoise4(0.1, 0.2, 0.3, 0.4, octaves=octaves),
             lambda: noise.snoise4(0.1, 0.2, 0.3, 0.4, octaves=octaves))):
        with pytest.raises(ValueError, match="Expected octaves value > 0"):
            call_()
        with pytest.raises(ValueError):
            reference_call()


def test_numpy_scalar_input():
    # NumPy scalars must take the scalar path and come back as floats.
    assert isinstance(mns.snoise4(np.float64(0.3), np.float64(0.7),
                                  np.float64(1.1), np.float64(2.2)), float)
    assert mns.snoise2(np.float64(0.3), np.float64(0.7)) == pytest.approx(
        noise.snoise2(0.3, 0.7), abs=ATOL)
    assert mns.snoise3(np.float32(0.3), 0.7, 1.1) == pytest.approx(
        noise.snoise3(0.3, 0.7, 1.1), abs=ATOL)
    assert isinstance(mns.snoise2(np.float64(0.3), np.float64(0.7)), float)
    assert isinstance(mns.snoise3(np.float32(0.3), 0.7, 1.1), float)


def test_output_range_is_normalised():
    xs, ys, zs, ws = coords(200_000, 25)
    for out in (mns.snoise2(xs, ys), mns.snoise3(xs, ys, zs),
                mns.snoise4(xs, ys, zs, ws)):
        assert out.min() > -1.05
        assert out.max() < 1.05


def test_determinism():
    # The permutation table is fixed, so the same point always gives the same
    # value regardless of how the field is driven or how often it is run.
    xs, ys, zs, _ = coords(2_000, 26)
    first = mns.snoise3(xs, ys, zs)
    assert np.array_equal(first, mns.snoise3(xs, ys, zs))
    assert np.array_equal(first, mns.snoise3(xs, ys, zs, octaves=1))
    for k in range(-3, 4):
        assert mns.snoise2(float(k), float(k)) == pytest.approx(
            noise.snoise2(float(k), float(k)), abs=ATOL)


def test_library_is_the_compiled_one():
    # `pixi run build` must produce the same library the bindings load, and it
    # must really have been loaded rather than stubbed.
    from mojo_noise_simplex import _lib
    path = _lib.build()
    assert os.path.exists(path)
    assert path.endswith(".so")
    assert os.path.getsize(path) > 0
    assert "mns_noise3_field" in dir(_lib.lib())