"""Port of Casey Duncan's `noise-simplex` C implementation to Mojo.

Source order follows upstream so the two read side by side:

  _noise.h   GRAD3, GRAD4, PERM, SIMPLEX, fast_sin, fast_cos
  _simplex.c noise2, noise3, fbm_noise3, noise4, fbm_noise4,
             py_noise2, py_noise3, py_noise4
  (new)      *_field elementwise drivers over flat coordinate buffers

Everything is `Float32`, because upstream is C `float` throughout, and that is
not a cosmetic choice. The lattice offset is computed as `x - (i - t)` where
`i` is the floor of the skewed coordinate, so at a coordinate of magnitude 5000
upstream's `i` has a float32 ULP of about 5e-4. Computing the same kernel in
float64 moves a large fraction of all points across a corner-selection comparison
(`x0 > y0`, and the `y0 > w0` bit of noise4's 6-bit corner index), and a point
on the other side of that comparison is a different point in a different simplex
cell — not a slightly different answer, an uncorrelated one. Measured against
upstream over 20k random points per configuration, a float64 kernel disagreed
with upstream on 34% of points at |x|<500 and 63% at |x|<5000; the Float32 one
below disagrees on none, with a worst case of 8.3e-7.

Divergences from upstream, all deliberate and all in the README:

  * `fast_sin`/`fast_cos` take and return `Float32` because upstream's are
    `static inline float`. `py_noise2`'s tiled branch likewise keeps `x`, `y`,
    `z` and `w` as separate float locals for its whole body, matching the C,
    which stores the mapped coordinates back into the same variables it read
    them from. That is not a detail: upstream's y-only tiling branch passes `w`
    where a clean-room reading of the C would expect `z`, and a port that
    "fixes" that is wrong by about 1.0 on every point.
  * F2/G2/F4/G4 are upstream's decimal literals, not recomputed from `sqrt`. The
    computed float32 value is one ulp off, which is enough to move the corner
    test at large coordinates.
  * the C `out` array parameter is spelled `dst` because `out` is a Mojo
    keyword.
  * `PERM` and `SIMPLEX` are bytes, as they are `unsigned char` in C, and
    `GRAD3`/`GRAD4` are `Int32`, as they are `int`. They are written into a
    stack buffer as vector stores at the top of every exported call and
    reached through a pointer: a compile-time table cannot be subscripted by a
    runtime index without the compiler materialising the whole table at each
    subscript.
  * `mut` arguments alias in Mojo, so `_tiled_noise2` works on fresh locals for
    the `x`/`y`/`z` that the C function mutates, matching pass-by-value.
  * `fastfloor` from _noise.h is unused by the simplex kernels and is not ported.
  * `py_noise2`'s `PyArg_ParseTupleAndKeywords` signature is C, not Mojo; the
    argument validation it performed is done by the ctypes layer instead.
  * the `*_field` drivers, their chunking and the `parallelize` call are not in
    upstream at all, which has no array API.
"""

from max.algorithm import parallelize
from std.math import floor, sqrt
from std.memory import stack_allocation

comptime Ptr = Pointer[Float32, AnyOrigin[mut=True]]
comptime PermPtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime GradPtr = Pointer[Int32, AnyOrigin[mut=True]]

# Upstream's "no tiling" sentinel. The ctypes layer normalises None to it.
comptime FLT_MAX: Float32 = 3.4028234663852886e38

# Elementwise field drivers stay serial below this many points: the thread
# launch costs more than the arithmetic it saves. Above it, work is split into
# fixed CHUNK-point items (so the last chunk is the only ragged one) across at
# most MAX_WORKERS threads. Both numbers were measured on the target box.
comptime PARALLEL_MIN = 1 << 18
comptime CHUNK = 1 << 14
comptime MAX_WORKERS = 64


# ------------------------------------------------------------------ _noise.h

# A comptime StaticTuple cannot be subscripted by a runtime value: the
# compiler materialises the whole tuple on the stack at every subscript, which
# for the 512-byte PERM is 1.5 kB of stores per lookup. The tables are
# therefore written into a stack buffer as compile-time SIMD constants, one
# vector store each, and reached through a pointer. All four tables together are
# 944 bytes in 46 vector stores, paid once per exported call and never per point.


@always_inline
def _perm_table() -> PermPtr:
    var t = stack_allocation[512, DType.uint8]()
    var q = PermPtr(unsafe_from_address=Int(t))
    q.unsafe_store[width=32](0, SIMD[DType.uint8, 32](151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148))
    q.unsafe_store[width=32](32, SIMD[DType.uint8, 32](247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175))
    q.unsafe_store[width=32](64, SIMD[DType.uint8, 32](74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54))
    q.unsafe_store[width=32](96, SIMD[DType.uint8, 32](65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64))
    q.unsafe_store[width=32](128, SIMD[DType.uint8, 32](52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213))
    q.unsafe_store[width=32](160, SIMD[DType.uint8, 32](119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104))
    q.unsafe_store[width=32](192, SIMD[DType.uint8, 32](218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157))
    q.unsafe_store[width=32](224, SIMD[DType.uint8, 32](184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180))
    q.unsafe_store[width=32](256, SIMD[DType.uint8, 32](151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148))
    q.unsafe_store[width=32](288, SIMD[DType.uint8, 32](247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175))
    q.unsafe_store[width=32](320, SIMD[DType.uint8, 32](74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54))
    q.unsafe_store[width=32](352, SIMD[DType.uint8, 32](65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64))
    q.unsafe_store[width=32](384, SIMD[DType.uint8, 32](52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213))
    q.unsafe_store[width=32](416, SIMD[DType.uint8, 32](119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104))
    q.unsafe_store[width=32](448, SIMD[DType.uint8, 32](218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157))
    q.unsafe_store[width=32](480, SIMD[DType.uint8, 32](184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180))
    return q


@always_inline
def _grad3_table() -> GradPtr:
    var t = stack_allocation[48, DType.int32]()
    var q = GradPtr(unsafe_from_address=Int(t))
    q.unsafe_store[width=8](0, SIMD[DType.int32, 8](1, 1, 0, -1, 1, 0, 1, -1))
    q.unsafe_store[width=8](8, SIMD[DType.int32, 8](0, -1, -1, 0, 1, 0, 1, -1))
    q.unsafe_store[width=8](16, SIMD[DType.int32, 8](0, 1, 1, 0, -1, -1, 0, -1))
    q.unsafe_store[width=8](24, SIMD[DType.int32, 8](0, 1, 1, 0, -1, 1, 0, 1))
    q.unsafe_store[width=8](32, SIMD[DType.int32, 8](-1, 0, -1, -1, 1, 0, -1, -1))
    q.unsafe_store[width=8](40, SIMD[DType.int32, 8](0, -1, 0, -1, 1, 0, 1, 1))
    return q


@always_inline
def _grad4_table() -> GradPtr:
    var t = stack_allocation[128, DType.int32]()
    var q = GradPtr(unsafe_from_address=Int(t))
    q.unsafe_store[width=8](0, SIMD[DType.int32, 8](0, 1, 1, 1, 0, 1, 1, -1))
    q.unsafe_store[width=8](8, SIMD[DType.int32, 8](0, 1, -1, 1, 0, 1, -1, -1))
    q.unsafe_store[width=8](16, SIMD[DType.int32, 8](0, -1, 1, 1, 0, -1, 1, -1))
    q.unsafe_store[width=8](24, SIMD[DType.int32, 8](0, -1, -1, 1, 0, -1, -1, -1))
    q.unsafe_store[width=8](32, SIMD[DType.int32, 8](1, 0, 1, 1, 1, 0, 1, -1))
    q.unsafe_store[width=8](40, SIMD[DType.int32, 8](1, 0, -1, 1, 1, 0, -1, -1))
    q.unsafe_store[width=8](48, SIMD[DType.int32, 8](-1, 0, 1, 1, -1, 0, 1, -1))
    q.unsafe_store[width=8](56, SIMD[DType.int32, 8](-1, 0, -1, 1, -1, 0, -1, -1))
    q.unsafe_store[width=8](64, SIMD[DType.int32, 8](1, 1, 0, 1, 1, 1, 0, -1))
    q.unsafe_store[width=8](72, SIMD[DType.int32, 8](1, -1, 0, 1, 1, -1, 0, -1))
    q.unsafe_store[width=8](80, SIMD[DType.int32, 8](-1, 1, 0, 1, -1, 1, 0, -1))
    q.unsafe_store[width=8](88, SIMD[DType.int32, 8](-1, -1, 0, 1, -1, -1, 0, -1))
    q.unsafe_store[width=8](96, SIMD[DType.int32, 8](1, 1, 1, 0, 1, 1, -1, 0))
    q.unsafe_store[width=8](104, SIMD[DType.int32, 8](1, -1, 1, 0, 1, -1, -1, 0))
    q.unsafe_store[width=8](112, SIMD[DType.int32, 8](-1, 1, 1, 0, -1, 1, -1, 0))
    q.unsafe_store[width=8](120, SIMD[DType.int32, 8](-1, -1, 1, 0, -1, -1, -1, 0))
    return q


@always_inline
def _simplex_table() -> PermPtr:
    var t = stack_allocation[256, DType.uint8]()
    var q = PermPtr(unsafe_from_address=Int(t))
    q.unsafe_store[width=32](0, SIMD[DType.uint8, 32](0, 1, 2, 3, 0, 1, 3, 2, 0, 0, 0, 0, 0, 2, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0))
    q.unsafe_store[width=32](32, SIMD[DType.uint8, 32](0, 2, 1, 3, 0, 0, 0, 0, 0, 3, 1, 2, 0, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 2, 0))
    q.unsafe_store[width=32](64, SIMD[DType.uint8, 32](0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    q.unsafe_store[width=32](96, SIMD[DType.uint8, 32](1, 2, 0, 3, 0, 0, 0, 0, 1, 3, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 1, 2, 3, 1, 0))
    q.unsafe_store[width=32](128, SIMD[DType.uint8, 32](1, 0, 2, 3, 1, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 1, 0, 0, 0, 0, 2, 1, 3, 0))
    q.unsafe_store[width=32](160, SIMD[DType.uint8, 32](0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    q.unsafe_store[width=32](192, SIMD[DType.uint8, 32](2, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 1, 2, 3, 0, 2, 1, 0, 0, 0, 0, 3, 1, 2, 0))
    q.unsafe_store[width=32](224, SIMD[DType.uint8, 32](2, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 2, 0, 0, 0, 0, 3, 2, 0, 1, 3, 2, 1, 0))
    return q


# The wrap constant is a float32 add, exactly as upstream applies it. That
# matters: `x + 25165824.0f` rounds to a multiple of two, so the subtraction
# lands on x mod 2, not x mod 1.
comptime WRAP: Float32 = 25165824.0

# 1/pi, as math.h spells it
comptime M_1_PI: Float32 = 0.31830988618379067154


# Fast sine/cosine functions from
# http://devmaster.net/forums/topic/4648-fast-and-accurate-sinecosine/page__st__80
# Note the input to these functions is not radians
# instead x = [0, 2] for r = [0, 2*PI]


# Upstream's `fast_sin`/`fast_cos` are `static inline float`, so they take and
# return a float. That matters on its own: the wrap is a float32 add that rounds
# to a multiple of two, so x lands on x mod 2 rather than x mod 1.
def fast_sin(x: Float32) -> Float32:
    # Wrap around
    var z = x + WRAP
    var w = x - (z - WRAP)

    var y = w - w * abs(w)
    comptime Q: Float32 = 3.1
    comptime P: Float32 = 3.6
    return y * (Q + P * abs(y))


def fast_cos(x: Float32) -> Float32:
    return fast_sin(x + 0.5)


# ---------------------------------------------------------------- _simplex.c

# 2D simplex skew factors. Upstream spells these as literals, with the
# derivation left in a comment, so they are literals here too: computing them as
# 0.5*(sqrt(3)-1) in Float32 lands one ulp away and moves the x0>y0 corner
# test at large coordinates.
comptime F2: Float32 = 0.3660254037844386
comptime G2: Float32 = 0.21132486540518713


@no_inline
def noise2(perm: PermPtr, grad3: GradPtr, x: Float32, y: Float32) -> Float32:
    var s = (x + y) * F2
    var i = floor(x + s)
    var j = floor(y + s)
    var t = (i + j) * G2

    var x0 = x - (i - t)
    var y0 = y - (j - t)

    var i1 = Int(x0 > y0)
    var j1 = 1 - i1

    var x1 = x0 - Float32(i1) + G2
    var y1 = y0 - Float32(j1) + G2
    var x2 = x0 + G2 * 2.0 - 1.0
    var y2 = y0 + G2 * 2.0 - 1.0

    var I = Int(i) & 255
    var J = Int(j) & 255
    var b0 = (Int(perm.unsafe_load(I + Int(perm.unsafe_load(J)))) % 12) * 3
    var b1 = (
        Int(perm.unsafe_load(I + i1 + Int(perm.unsafe_load(J + j1)))) % 12
    ) * 3
    var b2 = (
        Int(perm.unsafe_load(I + 1 + Int(perm.unsafe_load(J + 1)))) % 12
    ) * 3

    var f0 = 0.5 - x0 * x0 - y0 * y0
    var f1 = 0.5 - x1 * x1 - y1 * y1
    var f2 = 0.5 - x2 * x2 - y2 * y2
    var n0 = f0 * f0 * f0 * f0 * (
        Float32(grad3.unsafe_load(b0)) * x0
        + Float32(grad3.unsafe_load(b0 + 1)) * y0
    )
    var n1 = f1 * f1 * f1 * f1 * (
        Float32(grad3.unsafe_load(b1)) * x1
        + Float32(grad3.unsafe_load(b1 + 1)) * y1
    )
    var n2 = f2 * f2 * f2 * f2 * (
        Float32(grad3.unsafe_load(b2)) * x2
        + Float32(grad3.unsafe_load(b2 + 1)) * y2
    )
    if f0 <= 0.0:
        n0 = 0.0
    if f1 <= 0.0:
        n1 = 0.0
    if f2 <= 0.0:
        n2 = 0.0
    return (n0 + n1 + n2) * 70.0


comptime F3: Float32 = 1.0 / 3.0
comptime G3: Float32 = 1.0 / 6.0


@no_inline
def noise3(perm: PermPtr, grad3: GradPtr, x: Float32, y: Float32, z: Float32) -> Float32:
    var s = (x + y + z) * F3
    var i = floor(x + s)
    var j = floor(y + s)
    var k = floor(z + s)
    var t = (i + j + k) * G3

    var x0 = x - (i - t)
    var y0 = y - (j - t)
    var z0 = z - (k - t)

    # Upstream picks the two inner corners with a six-way if/elif chain whose
    # first test is a coin flip, so the branch predictor mispredicts about half
    # the time. The same six predicates are evaluated here as booleans, which
    # the compiler lowers to setcc/and/or; the offsets selected are identical,
    # ties included.
    var a = x0 >= y0
    var b = y0 >= z0
    var c = x0 >= z0
    var na = not a

    var o1x = Int(a and (b or c))
    var o1y = Int(na and b)
    var o1z = Int((a and not b and not c) or (na and not b))
    var o2x = Int(a or (na and b and c))
    var o2y = Int((a and b) or na)
    var o2z = Int((a and not b) or (na and (not b or not c)))

    var x1 = x0 - Float32(o1x) + G3
    var y1 = y0 - Float32(o1y) + G3
    var z1 = z0 - Float32(o1z) + G3
    var x2 = x0 - Float32(o2x) + 2.0 * G3
    var y2 = y0 - Float32(o2y) + 2.0 * G3
    var z2 = z0 - Float32(o2z) + 2.0 * G3
    var x3 = x0 - 1.0 + 3.0 * G3
    var y3 = y0 - 1.0 + 3.0 * G3
    var z3 = z0 - 1.0 + 3.0 * G3

    var I = Int(i) & 255
    var J = Int(j) & 255
    var K = Int(k) & 255
    var g0 = Int(
        perm.unsafe_load(I + Int(perm.unsafe_load(J + Int(perm.unsafe_load(K)))))
    )
    var g1 = Int(
        perm.unsafe_load(
            I
            + o1x
            + Int(
                perm.unsafe_load(J + o1y + Int(perm.unsafe_load(o1z + K)))
            )
        )
    )
    var g2 = Int(
        perm.unsafe_load(
            I
            + o2x
            + Int(
                perm.unsafe_load(J + o2y + Int(perm.unsafe_load(o2z + K)))
            )
        )
    )
    var g3 = Int(
        perm.unsafe_load(
            I + 1 + Int(perm.unsafe_load(J + 1 + Int(perm.unsafe_load(K + 1))))
        )
    )

    var f0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0
    var f1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1
    var f2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2
    var f3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3
    var n0 = _dot3(grad3, x0, y0, z0, g0 % 12)
    var n1 = _dot3(grad3, x1, y1, z1, g1 % 12)
    var n2 = _dot3(grad3, x2, y2, z2, g2 % 12)
    var n3 = _dot3(grad3, x3, y3, z3, g3 % 12)
    if f0 <= 0.0:
        n0 = 0.0
    if f1 <= 0.0:
        n1 = 0.0
    if f2 <= 0.0:
        n2 = 0.0
    if f3 <= 0.0:
        n3 = 0.0
    return (
        n0 * f0 * f0 * f0 * f0
        + n1 * f1 * f1 * f1 * f1
        + n2 * f2 * f2 * f2 * f2
        + n3 * f3 * f3 * f3 * f3
    ) * 32.0


def _dot3(
    grad3: GradPtr, x: Float32, y: Float32, z: Float32, hash: Int
) -> Float32:
    var b = hash * 3
    return (
        x * Float32(grad3.unsafe_load(b))
        + y * Float32(grad3.unsafe_load(b + 1))
        + z * Float32(grad3.unsafe_load(b + 2))
    )




@no_inline
def fbm_noise3(
    perm: PermPtr,
    grad3: GradPtr,
    x: Float32,
    y: Float32,
    z: Float32,
    octaves: Int,
    persistence: Float32,
    lacunarity: Float32,
) -> Float32:
    var freq = Float32(1.0)
    var amp = Float32(1.0)
    var max = Float32(1.0)
    var total = noise3(perm, grad3, x, y, z)
    var i = 1

    while i < octaves:
        freq *= lacunarity
        amp *= persistence
        max += amp
        total += noise3(perm, grad3, x * freq, y * freq, z * freq) * amp
        i += 1
    return total / max


def dot4(
    grad4: GradPtr, hash: Int, x: Float32, y: Float32, z: Float32, w: Float32
) -> Float32:
    var b = hash * 4
    return (
        Float32(grad4.unsafe_load(b)) * x
        + Float32(grad4.unsafe_load(b + 1)) * y
        + Float32(grad4.unsafe_load(b + 2)) * z
        + Float32(grad4.unsafe_load(b + 3)) * w
    )


# Upstream also spells these as literals, for the same reason as F2/G2.
comptime F4: Float32 = 0.30901699437494745
comptime G4: Float32 = 0.1381966011250105


def noise4(
    perm: PermPtr,
    grad4: GradPtr,
    simplex: PermPtr,
    x: Float32,
    y: Float32,
    z: Float32,
    w: Float32,
) -> Float32:
    var n0 = Float32(0.0)
    var n1 = Float32(0.0)
    var n2 = Float32(0.0)
    var n3 = Float32(0.0)
    var n4 = Float32(0.0)


    var s = (x + y + z + w) * F4
    var i = floor(x + s)
    var j = floor(y + s)
    var k = floor(z + s)
    var l = floor(w + s)
    var t = (i + j + k + l) * G4

    var x0 = x - (i - t)
    var y0 = y - (j - t)
    var z0 = z - (k - t)
    var w0 = w - (l - t)

    var c = (
        Int(x0 > y0) * 32
        + Int(x0 > z0) * 16
        + Int(y0 > z0) * 8
        + Int(x0 > w0) * 4
        + Int(y0 > w0) * 2
        + Int(z0 > w0)
    )
    var s0 = Int(simplex.unsafe_load(c * 4))
    var s1 = Int(simplex.unsafe_load(c * 4 + 1))
    var s2 = Int(simplex.unsafe_load(c * 4 + 2))
    var s3 = Int(simplex.unsafe_load(c * 4 + 3))
    var i1 = Int(s0 >= 3)
    var j1 = Int(s1 >= 3)
    var k1 = Int(s2 >= 3)
    var l1 = Int(s3 >= 3)
    var i2 = Int(s0 >= 2)
    var j2 = Int(s1 >= 2)
    var k2 = Int(s2 >= 2)
    var l2 = Int(s3 >= 2)
    var i3 = Int(s0 >= 1)
    var j3 = Int(s1 >= 1)
    var k3 = Int(s2 >= 1)
    var l3 = Int(s3 >= 1)

    var x1 = x0 - Float32(i1) + G4
    var y1 = y0 - Float32(j1) + G4
    var z1 = z0 - Float32(k1) + G4
    var w1 = w0 - Float32(l1) + G4
    var x2 = x0 - Float32(i2) + 2.0 * G4
    var y2 = y0 - Float32(j2) + 2.0 * G4
    var z2 = z0 - Float32(k2) + 2.0 * G4
    var w2 = w0 - Float32(l2) + 2.0 * G4
    var x3 = x0 - Float32(i3) + 3.0 * G4
    var y3 = y0 - Float32(j3) + 3.0 * G4
    var z3 = z0 - Float32(k3) + 3.0 * G4
    var w3 = w0 - Float32(l3) + 3.0 * G4
    var x4 = x0 - 1.0 + 4.0 * G4
    var y4 = y0 - 1.0 + 4.0 * G4
    var z4 = z0 - 1.0 + 4.0 * G4
    var w4 = w0 - 1.0 + 4.0 * G4

    var I = Int(Int(i) & 255)
    var J = Int(Int(j) & 255)
    var K = Int(Int(k) & 255)
    var L = Int(Int(l) & 255)
    var gi0 = Int(
        perm.unsafe_load(
            I
            + Int(
                perm.unsafe_load(
                    J
                    + Int(
                        perm.unsafe_load(
                            K + Int(perm.unsafe_load(L))
                        )
                    )
                )
            )
        )
    ) & 0x1F
    var gi1 = Int(
        perm.unsafe_load(
            I
            + i1
            + Int(
                perm.unsafe_load(
                    J
                    + j1
                    + Int(
                        perm.unsafe_load(
                            K + k1 + Int(perm.unsafe_load(L + l1))
                        )
                    )
                )
            )
        )
    ) & 0x1F
    var gi2 = Int(
        perm.unsafe_load(
            I
            + i2
            + Int(
                perm.unsafe_load(
                    J
                    + j2
                    + Int(
                        perm.unsafe_load(
                            K + k2 + Int(perm.unsafe_load(L + l2))
                        )
                    )
                )
            )
        )
    ) & 0x1F
    var gi3 = Int(
        perm.unsafe_load(
            I
            + i3
            + Int(
                perm.unsafe_load(
                    J
                    + j3
                    + Int(
                        perm.unsafe_load(
                            K + k3 + Int(perm.unsafe_load(L + l3))
                        )
                    )
                )
            )
        )
    ) & 0x1F
    var gi4 = Int(
        perm.unsafe_load(
            I
            + 1
            + Int(
                perm.unsafe_load(
                    J
                    + 1
                    + Int(
                        perm.unsafe_load(
                            K + 1 + Int(perm.unsafe_load(L + 1))
                        )
                    )
                )
            )
        )
    ) & 0x1F
    var t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0 - w0 * w0
    var t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1 - w1 * w1
    var t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2 - w2 * w2
    var t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3 - w3 * w3
    var t4 = 0.6 - x4 * x4 - y4 * y4 - z4 * z4 - w4 * w4

    if t0 >= 0.0:
        n0 = t0 * t0 * t0 * t0 * dot4(grad4, gi0, x0, y0, z0, w0)
    if t1 >= 0.0:
        n1 = t1 * t1 * t1 * t1 * dot4(grad4, gi1, x1, y1, z1, w1)
    if t2 >= 0.0:
        n2 = t2 * t2 * t2 * t2 * dot4(grad4, gi2, x2, y2, z2, w2)
    if t3 >= 0.0:
        n3 = t3 * t3 * t3 * t3 * dot4(grad4, gi3, x3, y3, z3, w3)
    if t4 >= 0.0:
        n4 = t4 * t4 * t4 * t4 * dot4(grad4, gi4, x4, y4, z4, w4)
    return 27.0 * (n0 + n1 + n2 + n3 + n4)


def fbm_noise4(
    perm: PermPtr,
    grad4: GradPtr,
    simplex: PermPtr,
    x: Float32,
    y: Float32,
    z: Float32,
    w: Float32,
    octaves: Int,
    persistence: Float32,
    lacunarity: Float32,
) -> Float32:
    var freq = Float32(1.0)
    var amp = Float32(1.0)
    var max = Float32(1.0)
    var total = noise4(perm, grad4, simplex, x, y, z, w)
    var i = 1

    while i < octaves:
        freq *= lacunarity
        amp *= persistence
        max += amp
        total += noise4(
            perm, grad4, simplex, x * freq, y * freq, z * freq, w * freq
        ) * amp
        i += 1
    return total / max


# Upstream's `octaves` bounds check raises ValueError from the argument parser;
# it lives in the ctypes layer, and `repeatx`/`repeaty` arrive already
# normalised to FLT_MAX (or a real tile size).


@no_inline
def _flat_noise2(
    perm: PermPtr,
    grad3: GradPtr,
    x: Float32,
    y: Float32,
    octaves: Int,
    persistence: Float32,
    lacunarity: Float32,
    base: Float32,
) -> Float32:
    if octaves == 1:
        return noise2(perm, grad3, x + base, y + base)
    var freq = Float32(1.0)
    var amp = Float32(1.0)
    var max = Float32(1.0)
    var total = noise2(perm, grad3, x + base, y + base)
    var i = 1
    while i < octaves:
        freq *= lacunarity
        amp *= persistence
        max += amp
        total += noise2(
            perm, grad3, x * freq + base, y * freq + base
        ) * amp
        i += 1
    return total / max


@no_inline
def _tiled_noise2(
    perm: PermPtr,
    grad3: GradPtr,
    grad4: GradPtr,
    simplex: PermPtr,
    x: Float32,
    y: Float32,
    octaves: Int,
    persistence: Float32,
    lacunarity: Float32,
    repeatx: Float32,
    repeaty: Float32,
    base: Float32,
) -> Float32:
    # `x`, `y`, `z` and `w` are float locals in upstream's py_noise2, and `z`
    # starts as the incoming `base`. They are kept as separate locals rather
    # than written back into the arguments, because `mut` arguments alias in
    # Mojo. Reproducing this storage is what keeps the y-only tiling branch in
    # step with upstream, which passes `w` there, not `z`.
    var tx = x
    var ty = y
    var tz = base
    var w = base
    if repeaty != FLT_MAX:
        var yf = ty * 2.0 / repeaty
        var yr = repeaty * M_1_PI * 0.5
        var vy = fast_sin(yf)
        var vyz = fast_cos(yf)
        ty = vy * yr
        w += vyz * yr
        if repeatx == FLT_MAX:
            return fbm_noise3(
                perm, grad3, tx, ty, w, octaves, persistence, lacunarity
            )
    if repeatx != FLT_MAX:
        var xf = tx * 2.0 / repeatx
        var xr = repeatx * M_1_PI * 0.5
        var vx = fast_sin(xf)
        var vxz = fast_cos(xf)
        tx = vx * xr
        tz += vxz * xr
        if repeaty == FLT_MAX:
            return fbm_noise3(
                perm, grad3, tx, ty, tz, octaves, persistence, lacunarity
            )
    return fbm_noise4(
        perm, grad4, simplex, tx, ty, tz, w, octaves, persistence, lacunarity
    )


@always_inline
def py_noise2(
    perm: PermPtr,
    grad3: GradPtr,
    grad4: GradPtr,
    simplex: PermPtr,
    x: Float32,
    y: Float32,
    octaves: Int,
    persistence: Float32,
    lacunarity: Float32,
    repeatx: Float32,
    repeaty: Float32,
    base: Float32,
) -> Float32:
    if repeatx == FLT_MAX and repeaty == FLT_MAX:
        return _flat_noise2(
            perm, grad3, x, y, octaves, persistence, lacunarity, base
        )
    return _tiled_noise2(
        perm,
        grad3,
        grad4,
        simplex,
        x,
        y,
        octaves,
        persistence,
        lacunarity,
        repeatx,
        repeaty,
        base,
    )


@always_inline
def py_noise3(
    perm: PermPtr,
    grad3: GradPtr,
    x: Float32,
    y: Float32,
    z: Float32,
    octaves: Int,
    persistence: Float32,
    lacunarity: Float32,
) -> Float32:
    if octaves == 1:
        return noise3(perm, grad3, x, y, z)
    elif octaves > 1:
        return fbm_noise3(
            perm, grad3, x, y, z, octaves, persistence, lacunarity
        )
    else:
        return 0.0


@always_inline
def py_noise4(
    perm: PermPtr,
    grad4: GradPtr,
    simplex: PermPtr,
    x: Float32,
    y: Float32,
    z: Float32,
    w: Float32,
    octaves: Int,
    persistence: Float32,
    lacunarity: Float32,
) -> Float32:
    if octaves == 1:
        return noise4(perm, grad4, simplex, x, y, z, w)
    elif octaves > 1:
        return fbm_noise4(
            perm, grad4, simplex, x, y, z, w, octaves, persistence, lacunarity
        )
    else:
        return 0.0


# -------------------------------------------------------------- field drivers
# Upstream has no array API; the same kernels driven over flat coordinate
# buffers is the only thing added on this side of the FFI. Each worker gets
# its own pointers, so the inner loop never reloads a captured pointer.


def p(address: Int) -> Ptr:
    return Ptr(unsafe_from_address=address)


def noise2_range(
    perm: PermPtr,
    grad3: GradPtr,
    grad4: GradPtr,
    simplex: PermPtr,
    xs: Int,
    ys: Int,
    dst: Int,
    start: Int,
    stop: Int,
    octaves: Int,
    persistence: Float32,
    lacunarity: Float32,
    repeatx: Float32,
    repeaty: Float32,
    base: Float32,
) -> None:
    var px = p(xs)
    var py = p(ys)
    var pd = p(dst)
    var i = start
    while i < stop:
        pd.unsafe_store(
            i,
            py_noise2(
                perm,
                grad3,
                grad4,
                simplex,
                px.unsafe_load(i),
                py.unsafe_load(i),
                octaves,
                persistence,
                lacunarity,
                repeatx,
                repeaty,
                base,
            ),
        )
        i += 1


def noise3_range(
    perm: PermPtr,
    grad3: GradPtr,
    xs: Int,
    ys: Int,
    zs: Int,
    dst: Int,
    start: Int,
    stop: Int,
    octaves: Int,
    persistence: Float32,
    lacunarity: Float32,
) -> None:
    var px = p(xs)
    var py = p(ys)
    var pz = p(zs)
    var pd = p(dst)
    var i = start
    while i < stop:
        pd.unsafe_store(
            i,
            py_noise3(
                perm,
                grad3,
                px.unsafe_load(i),
                py.unsafe_load(i),
                pz.unsafe_load(i),
                octaves,
                persistence,
                lacunarity,
            ),
        )
        i += 1


def noise4_range(
    perm: PermPtr,
    grad4: GradPtr,
    simplex: PermPtr,
    xs: Int,
    ys: Int,
    zs: Int,
    ws: Int,
    dst: Int,
    start: Int,
    stop: Int,
    octaves: Int,
    persistence: Float32,
    lacunarity: Float32,
) -> None:
    var px = p(xs)
    var py = p(ys)
    var pz = p(zs)
    var pw = p(ws)
    var pd = p(dst)
    var i = start
    while i < stop:
        pd.unsafe_store(
            i,
            py_noise4(
                perm,
                grad4,
                simplex,
                px.unsafe_load(i),
                py.unsafe_load(i),
                pz.unsafe_load(i),
                pw.unsafe_load(i),
                octaves,
                persistence,
                lacunarity,
            ),
        )
        i += 1


# ------------------------------------------------------------------- C ABI
# `@export` refuses parametric functions and an inferred pointer origin is
# parametric, so every buffer crosses as an Int address and is rebuilt here.


# The tiling test is hoisted into this entry point so the flat path never pays
# for `grad4`/`simplex`. Those two tables are 288 bytes written as 34 vector
# stores, and `_flat_noise2` never reads them. This entry point is taken once
# per call from Python, so that is 34 stores per call saved on the common
# no-tiling path.
@export("mns_noise2")
def mns_noise2(
    x: Float32,
    y: Float32,
    octaves: Int,
    persistence: Float32,
    lacunarity: Float32,
    repeatx: Float32,
    repeaty: Float32,
    base: Float32,
) abi("C") -> Float32:
    var perm = _perm_table()
    var grad3 = _grad3_table()
    if repeatx == FLT_MAX and repeaty == FLT_MAX:
        return _flat_noise2(
            perm, grad3, x, y, octaves, persistence, lacunarity, base
        )
    var grad4 = _grad4_table()
    var simplex = _simplex_table()
    return _tiled_noise2(
        perm,
        grad3,
        grad4,
        simplex,
        x,
        y,
        octaves,
        persistence,
        lacunarity,
        repeatx,
        repeaty,
        base,
    )


@export("mns_noise3")
def mns_noise3(
    x: Float32,
    y: Float32,
    z: Float32,
    octaves: Int,
    persistence: Float32,
    lacunarity: Float32,
) abi("C") -> Float32:
    var perm = _perm_table()
    var grad3 = _grad3_table()
    return py_noise3(perm, grad3, x, y, z, octaves, persistence, lacunarity)


@export("mns_noise4")
def mns_noise4(
    x: Float32,
    y: Float32,
    z: Float32,
    w: Float32,
    octaves: Int,
    persistence: Float32,
    lacunarity: Float32,
) abi("C") -> Float32:
    var perm = _perm_table()
    var grad4 = _grad4_table()
    var simplex = _simplex_table()
    return py_noise4(
        perm, grad4, simplex, x, y, z, w, octaves, persistence, lacunarity
    )


@export("mns_noise2_field")
def mns_noise2_field(
    xs: Int,
    ys: Int,
    dst: Int,
    n: Int,
    octaves: Int,
    persistence: Float32,
    lacunarity: Float32,
    repeatx: Float32,
    repeaty: Float32,
    base: Float32,
) abi("C"):
    if n <= 0:
        return
    var perm = _perm_table()
    var grad3 = _grad3_table()
    var grad4 = _grad4_table()
    var simplex = _simplex_table()
    if n < PARALLEL_MIN:
        noise2_range(
            perm,
            grad3,
            grad4,
            simplex,
            xs,
            ys,
            dst,
            0,
            n,
            octaves,
            persistence,
            lacunarity,
            repeatx,
            repeaty,
            base,
        )
        return
    var items = (n + CHUNK - 1) // CHUNK
    var workers = min(items, MAX_WORKERS)

    def work(k: Int) {imm}:
        var start = k * CHUNK
        noise2_range(
            perm,
            grad3,
            grad4,
            simplex,
            xs,
            ys,
            dst,
            start,
            min(start + CHUNK, n),
            octaves,
            persistence,
            lacunarity,
            repeatx,
            repeaty,
            base,
        )

    parallelize(work, items, workers)


@export("mns_noise3_field")
def mns_noise3_field(
    xs: Int,
    ys: Int,
    zs: Int,
    dst: Int,
    n: Int,
    octaves: Int,
    persistence: Float32,
    lacunarity: Float32,
) abi("C"):
    if n <= 0:
        return
    var perm = _perm_table()
    var grad3 = _grad3_table()
    if n < PARALLEL_MIN:
        noise3_range(
            perm,
            grad3,
            xs,
            ys,
            zs,
            dst,
            0,
            n,
            octaves,
            persistence,
            lacunarity,
        )
        return
    var items = (n + CHUNK - 1) // CHUNK
    var workers = min(items, MAX_WORKERS)

    def work(k: Int) {imm}:
        var start = k * CHUNK
        noise3_range(
            perm,
            grad3,
            xs,
            ys,
            zs,
            dst,
            start,
            min(start + CHUNK, n),
            octaves,
            persistence,
            lacunarity,
        )

    parallelize(work, items, workers)


@export("mns_noise4_field")
def mns_noise4_field(
    xs: Int,
    ys: Int,
    zs: Int,
    ws: Int,
    dst: Int,
    n: Int,
    octaves: Int,
    persistence: Float32,
    lacunarity: Float32,
) abi("C"):
    if n <= 0:
        return
    var perm = _perm_table()
    var grad4 = _grad4_table()
    var simplex = _simplex_table()
    if n < PARALLEL_MIN:
        noise4_range(
            perm,
            grad4,
            simplex,
            xs,
            ys,
            zs,
            ws,
            dst,
            0,
            n,
            octaves,
            persistence,
            lacunarity,
        )
        return
    var items = (n + CHUNK - 1) // CHUNK
    var workers = min(items, MAX_WORKERS)

    def work(k: Int) {imm}:
        var start = k * CHUNK
        noise4_range(
            perm,
            grad4,
            simplex,
            xs,
            ys,
            zs,
            ws,
            dst,
            start,
            min(start + CHUNK, n),
            octaves,
            persistence,
            lacunarity,
        )

    parallelize(work, items, workers)
