"""Direct tests of the compiled shared library, one per exported kernel.

Every symbol in `src/ported.mojo` that carries `@export` is called here by
name, through `ctypes`, and asserted on its actual return value. These are the
tests that make the kernels measured rather than merely reachable: the rest of
the suite could in principle be satisfied by a Python reimplementation, these
cannot.

The conftest hooks rebuild the library and fail the run if it is not there, so
a "pass" here means the compiled kernel produced these numbers.
"""

import os

import numpy as np
import pytest

import mojo_noise_simplex as mns
from mojo_noise_simplex import _lib

noise = pytest.importorskip("noise")

FM = mns.FLT_MAX

# Upstream computes in C float; this port does too, and the Python layer
# narrows inputs to float32 exactly as upstream's PyArg_ParseTuple does. What
# is left is pure float32 arithmetic noise plus the occasional point that lands
# on the far side of a corner-selection comparison, which is a different point
# in a different simplex cell. Both are bounded far below this.
ATOL = 1e-5


def call(symbol, *args):
    """Invoke one exported kernel directly and return its raw result."""
    return getattr(_lib.lib(), symbol)(
        *[a if isinstance(a, int) else float(a) for a in args])


def call_field(symbol, columns, *args):
    """Drive an exported field kernel over flat float32 coordinate buffers."""
    n = len(columns[0])
    flat = [np.ascontiguousarray(c, dtype=np.float32) for c in columns]
    out = np.empty(n, dtype=np.float32)
    addrs = [c.ctypes.data for c in flat] + [out.ctypes.data, n]
    getattr(_lib.lib(), symbol)(
        *addrs, *[a if isinstance(a, int) else float(a) for a in args])
    return out


# The list below is the contract: every @export in src/ported.mojo, spelled out.
# The first test in this file asserts it matches the source, so a kernel added
# to the .mojo file without a test here fails the suite.
EXPORTED = [
    "mns_noise2",
    "mns_noise3",
    "mns_noise4",
    "mns_noise2_field",
    "mns_noise3_field",
    "mns_noise4_field",
]


def test_every_export_is_covered_by_this_file():
    """Every @export symbol in src/ported.mojo is named in this module."""
    import re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    found = set()
    for dirpath, _, names in os.walk(os.path.join(root, "src")):
        for name in names:
            if not name.endswith(".mojo"):
                continue
            with open(os.path.join(dirpath, name)) as fh:
                found |= set(re.findall(r'@export\("([A-Za-z_0-9]+)"\)', fh.read()))
    assert found == set(EXPORTED), (
        f"src/ exports {sorted(found)} but this file names {sorted(EXPORTED)}")


def test_all_exports_are_present_in_the_library():
    handle = _lib.lib()
    for symbol in EXPORTED:
        assert hasattr(handle, symbol), f"{symbol} missing from {_lib.lib_path()}"


# ------------------------------------------------------------------ scalars

SCALAR_POINTS = [(0.0, 0.0), (0.5, 0.5), (0.3, 0.7), (-1.25, 4.5),
                 (17.0, -9.0), (1e4, -1e4), (0.123456789, 0.987654321)]


@pytest.mark.parametrize("x,y", SCALAR_POINTS)
def test_mns_noise2(x, y):
    # 1 octave, no tiling: the argument list is upstream's, in upstream's order.
    assert call("mns_noise2", x, y, 1, 0.5, 2.0, FM, FM, 0.0) == pytest.approx(
        noise.snoise2(x, y), abs=ATOL)


@pytest.mark.parametrize("octaves", [1, 2, 5])
@pytest.mark.parametrize("base", [0.0, 3.25])
def test_mns_noise2_octaves_and_base(octaves, base):
    for x, y in SCALAR_POINTS:
        assert call("mns_noise2", x, y, octaves, 0.5, 2.0, FM, FM, base) == \
            pytest.approx(noise.snoise2(x, y, octaves=octaves, base=base), abs=ATOL)


@pytest.mark.parametrize("repeatx,repeaty", [(8.0, 8.0), (8.0, FM), (FM, 8.0),
                                             (4.0, 16.0), (25.0, FM)])
@pytest.mark.parametrize("octaves", [1, 4])
def test_mns_noise2_tiled(repeatx, repeaty, octaves):
    # The three tiling branches end in different kernels (fbm_noise4, and
    # fbm_noise3 reached via x or via y), so all three are covered here. The
    # y-only case is the one that catches upstream's `w`-not-`z` argument.
    for x, y in SCALAR_POINTS:
        assert call("mns_noise2", x, y, octaves, 0.5, 2.0, repeatx, repeaty,
                    0.0) == pytest.approx(
            noise.snoise2(x, y, octaves=octaves, repeatx=repeatx,
                          repeaty=repeaty), abs=ATOL)


@pytest.mark.parametrize("x,y,z", [(0.0, 0.0, 0.0), (0.3, 0.7, 1.1),
                                   (-1.25, 4.5, -6.75), (17.0, -9.0, 3.5),
                                   (5000.0, -3000.0, 12.5)])
def test_mns_noise3(x, y, z):
    assert call("mns_noise3", x, y, z, 1, 0.5, 2.0) == pytest.approx(
        noise.snoise3(x, y, z), abs=ATOL)


@pytest.mark.parametrize("octaves,persistence,lacunarity",
                         [(1, 0.5, 2.0), (2, 0.5, 2.0), (6, 0.5, 2.0),
                          (5, 0.9, 1.7), (4, 0.1, 2.0), (9, 0.5, 3.5)])
def test_mns_noise3_fbm(octaves, persistence, lacunarity):
    for x, y, z in [(0.3, 0.7, 1.1), (-4.5, 9.25, -6.75), (60.0, -12.0, 3.5)]:
        assert call("mns_noise3", x, y, z, octaves, persistence, lacunarity) == \
            pytest.approx(
                noise.snoise3(x, y, z, octaves=octaves,
                              persistence=persistence,
                              lacunarity=lacunarity), abs=ATOL)


@pytest.mark.parametrize("x,y,z,w", [(0.0, 0.0, 0.0, 0.0), (0.3, 0.7, 1.1, 2.2),
                                     (-1.25, 4.5, -6.75, 0.125),
                                     (5.0, 5.0, 5.0, 5.0),
                                     (2000.0, -2000.0, 0.5, -0.5)])
def test_mns_noise4(x, y, z, w):
    assert call("mns_noise4", x, y, z, w, 1, 0.5, 2.0) == pytest.approx(
        noise.snoise4(x, y, z, w), abs=ATOL)


@pytest.mark.parametrize("octaves,persistence", [(1, 0.5), (3, 0.5), (9, 0.5),
                                                (4, 0.9), (4, 0.1)])
def test_mns_noise4_fbm(octaves, persistence):
    # Upstream's snoise4 segfaults if `lacunarity` is passed, so it is never
    # passed here either; upstream's own default of 2.0 is used on both sides.
    for x, y, z, w in [(0.3, 0.7, 1.1, 2.2), (-4.5, 9.25, -6.75, 0.125)]:
        assert call("mns_noise4", x, y, z, w, octaves, persistence, 2.0) == \
            pytest.approx(
                noise.snoise4(x, y, z, w, octaves=octaves,
                              persistence=persistence), abs=ATOL)


def test_mns_noise3_takes_both_the_simple_and_the_fbm_branch():
    # octaves == 1 must take the simple-noise branch and octaves > 1 the fBm
    # one. A kernel that always took a single branch would still pass a
    # one-octave test, so both are checked, and the two must differ.
    one = call("mns_noise3", 0.3, 0.7, 1.1, 1, 0.5, 2.0)
    many = call("mns_noise3", 0.3, 0.7, 1.1, 7, 0.5, 2.0)
    assert one == pytest.approx(noise.snoise3(0.3, 0.7, 1.1), abs=ATOL)
    assert many == pytest.approx(noise.snoise3(0.3, 0.7, 1.1, octaves=7), abs=ATOL)
    assert abs(one - many) > 1e-3


def test_mns_noise4_takes_both_the_simple_and_the_fbm_branch():
    one = call("mns_noise4", 0.3, 0.7, 1.1, 2.2, 1, 0.5, 2.0)
    many = call("mns_noise4", 0.3, 0.7, 1.1, 2.2, 6, 0.5, 2.0)
    assert one == pytest.approx(noise.snoise4(0.3, 0.7, 1.1, 2.2), abs=ATOL)
    assert many == pytest.approx(noise.snoise4(0.3, 0.7, 1.1, 2.2, octaves=6),
                                 abs=ATOL)
    assert abs(one - many) > 1e-3


# ------------------------------------------------------------------ fields

def _coords(n, seed, lo=-20.0, hi=20.0, k=4):
    """float32 coordinates, the type upstream's argument parser narrows to."""
    rng = np.random.default_rng(seed)
    return [rng.uniform(lo, hi, n).astype(np.float32) for _ in range(k)]


def _upstream_series(fn, columns):
    return np.array([fn(*row) for row in zip(*columns)], dtype=np.float64)


def _agree(ours, theirs):
    ours = np.asarray(ours, np.float64)
    theirs = np.asarray(theirs, np.float64)
    diff = np.abs(ours - theirs)
    over = diff > ATOL
    assert over.mean() <= 1e-3, (
        f"{over.mean():.4%} of points disagree by more than {ATOL}; "
        f"worst {diff.max():.3e}")
    assert diff.max() <= 1e-3, f"worst disagreement {diff.max():.3e}"


def test_mns_noise2_field_serial():
    xs, ys = _coords(5_000, 1)[:2]
    _agree(call_field("mns_noise2_field", [xs, ys], 1, 0.5, 2.0, FM, FM, 0.0),
           _upstream_series(lambda a, b: noise.snoise2(a, b), [xs, ys]))


def test_mns_noise2_field_octaves_and_base():
    xs, ys = _coords(5_000, 2)[:2]
    _agree(call_field("mns_noise2_field", [xs, ys], 4, 0.5, 2.0, FM, FM, 1.75),
           _upstream_series(
               lambda a, b: noise.snoise2(a, b, octaves=4, base=1.75),
               [xs, ys]))


@pytest.mark.parametrize("repeatx,repeaty", [(8.0, 8.0), (8.0, FM), (FM, 8.0),
                                             (5.0, 12.0), (25.0, 8.0)])
def test_mns_noise2_field_tiled(repeatx, repeaty):
    xs, ys = _coords(5_000, 3)[:2]
    _agree(call_field("mns_noise2_field", [xs, ys], 3, 0.5, 2.0, repeatx, repeaty,
                      0.0),
           _upstream_series(
               lambda a, b: noise.snoise2(a, b, octaves=3, repeatx=repeatx,
                                          repeaty=repeaty), [xs, ys]))


def test_mns_noise3_field_serial():
    xs, ys, zs, _ = _coords(5_000, 4)
    _agree(call_field("mns_noise3_field", [xs, ys, zs], 1, 0.5, 2.0),
           _upstream_series(lambda a, b, c: noise.snoise3(a, b, c),
                            [xs, ys, zs]))


def test_mns_noise3_field_fbm():
    xs, ys, zs, _ = _coords(5_000, 5)
    _agree(call_field("mns_noise3_field", [xs, ys, zs], 5, 0.5, 2.0),
           _upstream_series(
               lambda a, b, c: noise.snoise3(a, b, c, octaves=5),
               [xs, ys, zs]))


def test_mns_noise4_field_serial():
    xs, ys, zs, ws = _coords(5_000, 6)
    _agree(call_field("mns_noise4_field", [xs, ys, zs, ws], 1, 0.5, 2.0),
           _upstream_series(lambda a, b, c, d: noise.snoise4(a, b, c, d),
                            [xs, ys, zs, ws]))


def test_mns_noise4_field_fbm():
    xs, ys, zs, ws = _coords(5_000, 7)
    _agree(call_field("mns_noise4_field", [xs, ys, zs, ws], 4, 0.5, 2.0),
           _upstream_series(
               lambda a, b, c, d: noise.snoise4(a, b, c, d, octaves=4),
               [xs, ys, zs, ws]))


@pytest.mark.parametrize("lo,hi", [(-20.0, 20.0), (-500.0, 500.0)])
def test_field_kernels_hold_at_large_coordinates(lo, hi):
    # The lattice offset's float32 ULP grows with the coordinate, so this is
    # where a kernel computed at the wrong width falls apart. A float64 kernel
    # disagrees with upstream on ~30-45% of points in this range.
    xs, ys, zs, ws = _coords(4_000, 11, lo, hi)
    for coords, symbol, tail, ref in (
            ([xs, ys], "mns_noise2_field", (1, 0.5, 2.0, FM, FM, 0.0),
             lambda a, b: noise.snoise2(a, b)),
            ([xs, ys, zs], "mns_noise3_field", (1, 0.5, 2.0),
             lambda a, b, c: noise.snoise3(a, b, c)),
            ([xs, ys, zs, ws], "mns_noise4_field", (1, 0.5, 2.0),
             lambda a, b, c, d: noise.snoise4(a, b, c, d))):
        _agree(call_field(symbol, coords, *tail), _upstream_series(ref, coords))


@pytest.mark.parametrize("n", [1, 2, 17, 4_096])
def test_field_kernels_handle_short_inputs(n):
    # Every worker takes a fixed 16384-point chunk, so a short field is where a
    # bounds mistake in the chunk arithmetic shows up.
    xs, ys, zs, ws = _coords(max(n, 1), 8)
    assert call_field("mns_noise2_field", [xs[:n], ys[:n]], 1, 0.5, 2.0, FM, FM,
                      0.0).size == n
    assert call_field("mns_noise3_field", [xs[:n], ys[:n], zs[:n]], 1, 0.5,
                      2.0).size == n
    assert call_field("mns_noise4_field",
                      [xs[:n], ys[:n], zs[:n], ws[:n]], 1, 0.5, 2.0).size == n


def test_field_kernels_handle_zero_length_inputs():
    empty = np.empty(0, dtype=np.float32)
    for symbol, cols, tail in (("mns_noise2_field", [empty, empty],
                                (FM, FM, 0.0)),
                               ("mns_noise3_field", [empty] * 3, (2.0,)),
                               ("mns_noise4_field", [empty] * 4, (2.0,))):
        # call_field supplies the 4 kernel-specific float arguments itself;
        # `tail` is the extra 2D tiling sentinel pair only.
        extra = (FM, FM, 0.0) if symbol == "mns_noise2_field" else ()
        assert call_field(symbol, cols, 1, 0.5, 2.0, *extra).size == 0


@pytest.mark.parametrize("n", [262_144, 262_145, 300_000])
def test_threaded_field_kernels_are_bit_identical_to_their_serial_prefix(n):
    # PARALLEL_MIN is 262144 points, so these cross into the threaded driver.
    # Chunking must not change a single bit of the answer.
    xs, ys, zs, _ = _coords(n, 9)
    ws = np.linspace(-9.0, 9.0, n).astype(np.float32)
    prefix = 2_048

    def head(columns):
        return [np.ascontiguousarray(c[:prefix]) for c in columns]

    for columns, symbol, tail in (
            ([xs, ys], "mns_noise2_field", (1, 0.5, 2.0, FM, FM, 0.0)),
            ([xs, ys], "mns_noise2_field", (4, 0.5, 2.0, 8.0, 8.0, 0.0)),
            ([xs, ys, zs], "mns_noise3_field", (1, 0.5, 2.0)),
            ([xs, ys, zs, ws], "mns_noise4_field", (3, 0.5, 2.0))):
        threaded = call_field(symbol, columns, *tail)
        serial = call_field(symbol, head(columns), *tail)
        assert np.array_equal(threaded[:prefix], serial), (
            f"{symbol} at n={n}: threaded prefix differs from serial")


def test_field_kernels_write_only_the_first_n_entries():
    # A kernel that wrote past `n`, or into its own copy, would still return the
    # right length. Check the destination buffer itself.
    n = 300
    xs, ys = _coords(n, 10)[:2]
    out = np.full(n + 20, -12345.0, dtype=np.float32)
    _lib.lib().mns_noise2_field(xs.ctypes.data, ys.ctypes.data, out.ctypes.data,
                                n, 1, 0.5, 2.0, FM, FM, 0.0)
    assert np.all(out[n:] == -12345.0)
    assert np.all(np.isfinite(out[:n]))
    assert not np.all(out[:n] == 0.0)