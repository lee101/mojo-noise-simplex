# mojo-noise-simplex

[Casey Duncan's `noise-simplex`](https://github.com/caseman/noise) kernels,
ported to [Mojo](https://www.modular.com/mojo) and callable from Python with the
same names, argument order and defaults.

```python
import numpy as np
import mojo_noise_simplex as mns

mns.snoise2(0.3, 0.7)                          # float, like noise.snoise2
mns.snoise2(xs, ys)                            # a whole 2D field, broadcasting
mns.snoise3(xs, ys, zs, octaves=6)             # 6-octave fBm
mns.snoise2(xs, ys, repeatx=8, repeaty=8)      # tileable field
```

Upstream is scalar-only: its C entry points take one point at a time, so a
field is a Python loop. That loop is what this port replaces, with results that
match upstream to float32 precision rather than merely resembling them.

## Install

```bash
pixi install     # brings its own Mojo toolchain, builds nothing
pixi run build   # src/ported.mojo -> dist/libmojo-noise-simplex.so
pixi run test    # 127 parity tests against upstream `noise`
pixi run bench   # benchmark against upstream `noise`
```

`pixi run test` builds the library if `dist/` is stale, so `pixi install && pixi
run test` is enough. Outside pixi, put `python/` on `PYTHONPATH` with `mojo` (and
the `max` package) on `PATH`; the package builds the shared library on import.

## Usage

```python
import numpy as np
import mojo_noise_simplex as mns

# a 512x512 height field, tiled so it repeats every 32 units
gx, gy = np.meshgrid(np.linspace(0, 32, 512), np.linspace(0, 32, 512))
field = mns.snoise2(gx, gy, octaves=4, repeatx=32.0, repeaty=32.0)
print(field.shape, field.min(), field.max())     # (512, 512) -0.789 ... 0.76

# 3D and 4D, and the fBm arguments
vol = mns.snoise3(gx[..., None], gy[..., None], np.zeros((512, 512, 1)),
                  octaves=6, persistence=0.5, lacunarity=2.0)
print(mns.snoise4(0.3, 0.7, 1.1, 2.2, octaves=5))   # 0.18848596513271332
```

`repeatx`/`repeaty` accept `None` as well as `FLT_MAX` for "no tiling", matching
upstream's docstring spelling. Coordinates may be scalars (returning a `float`)
or any broadcastable arrays (returning an `ndarray`); mixing the two works, and
the output has the broadcast shape.

## What is covered

| upstream | status |
| --- | --- |
| `noise2` | ported |
| `noise3` | ported |
| `fbm_noise3` | ported |
| `noise4` | ported |
| `fbm_noise4` | ported |
| `py_noise2` (incl. tiled `repeatx`/`repeaty`/`base`) | ported |
| `py_noise3` | ported |
| `py_noise4` | ported |
| `fast_sin`, `fast_cos` | ported |
| `GRAD3`, `GRAD4`, `PERM`, `SIMPLEX` | ported, verified byte-for-byte |
| `fastfloor` | not ported; unused by the simplex kernels |
| `PyArg_ParseTupleAndKeywords` argument parsing | moved to the ctypes layer |
| `pnoise1`, `pnoise2`, `pnoise3` (Perlin) | not in scope |

`src/ported.mojo` follows `_simplex.c` and `_noise.h` top to bottom: the tables
in the order they are declared, then `noise2`, `noise3`, `fbm_noise3`,
`noise4`, `fbm_noise4`, `py_noise2`, `py_noise3`, `py_noise4`, and only then the
added field drivers. The two files read side by side.

### Fidelity, and why it is float32

Everything is `Float32`, because upstream is C `float` throughout. That is not
cosmetic, and it is the main thing this port had to get right.

The lattice offset is `x - (i - t)`, where `i` is the floor of the skewed
coordinate. At a coordinate of magnitude 5000, upstream's `i` has a float32 ULP
of about 5e-4. A float64 kernel puts the offset on the other side of the
`x0 > y0` corner test for a large fraction of all points, and a point on the
other side of that comparison is a point in a different simplex cell: not a
slightly different answer, an uncorrelated one. Measured against upstream over
20k random points, a float64 version of this same kernel disagreed with it on
**34% of points at |x| < 500 and 63% at |x| < 5000**. The float32 version
disagrees on **none**, with a worst observed error of 8.3e-7 — one float32 ulp
at 0.5.

So the port is deliberately not "more accurate than upstream". It is the same
arithmetic at the same width. The Python layer narrows every coordinate to
float32 before the call, exactly as upstream's `PyArg_ParseTuple` format string
does.

Two consequences worth knowing:

- `F2`, `G2`, `F4` and `G4` are upstream's decimal literals, not recomputed
  from `sqrt`. The computed float32 value is one ulp off, which is enough to
  move the corner test at large coordinates.
- Upstream's y-only tiling branch passes `w` where a clean-room reading of the C
  would expect `z`. The port passes `w`, because that is what upstream does; a
  transcription that "fixes" it disagrees by about 1.0 on every point.

### Documented divergences

- **Width**: float32, as upstream. Not a divergence, but stated because it looks
  like one in a Mojo port and is the opposite of what you would guess.
- **Tables on the stack**: `PERM`, `SIMPLEX`, `GRAD3` and `GRAD4` are written
  into a stack buffer as vector stores at the top of each exported call and
  reached through a pointer. A `comptime StaticTuple` cannot be subscripted by a
  runtime index without the compiler materialising the whole table at every
  subscript — for the 512-byte `PERM` that is 1.5 kB of stores per lookup. The
  values are identical; only the storage differs. Mojo has no module-level
  variables on this toolchain (`global variables are not supported`), so there
  is nowhere to cache them between calls and they must be rebuilt per call.
  That is also why the builders cannot be marked `@no_inline`: a function
  returning a `stack_allocation` pointer has to be inlined, or the caller's
  frame is never reserved and every result comes back wrong. Verified on this
  toolchain, not assumed.
- **Field drivers**: the `*_field` kernels, their chunking and the `parallelize`
  call are additions. Upstream has no array API at all, so there is nothing to
  mirror. Their per-point body is upstream's, unchanged.
- **Argument parsing**: upstream validates in C via `PyArg_ParseTuple`. That
  validation (`octaves > 0`) happens in `simplex.py` here.
- **`noise3`'s corner selection**: upstream picks the two inner corners with a
  six-way `if`/`elif` chain. The same six predicates are evaluated as booleans
  and combined, which the compiler lowers to `setcc`/`and`/`or`. The offsets
  selected are identical, ties included; only the branch structure differs.

## Benchmarks

`pixi run bench` on an Intel Xeon E5-2697 v4 @ 2.30GHz, 2 sockets x 18 cores x
2 threads (72 logical), Linux x86_64, Python 3.13.14, NumPy 2.5.1:

| case | mojo-noise-simplex | upstream `noise` | |
| --- | ---: | ---: | --- |
| snoise2 scalar x 200k | 509.7 ms | 100.2 ms | 5.1x slower |
| snoise2 field, 4M points | 39.2 ms | 2408.5 ms | **61.4x faster** |
| snoise3 field, 4M points | 57.2 ms | 1920.2 ms | **33.6x faster** |
| snoise4 field, 1M points | 19.7 ms | 708.7 ms | **35.9x faster** |
| snoise3 field, 4M points, 6 octaves | 108.5 ms | 4729.1 ms | **43.6x faster** |
| snoise2 field, 4M points, tiled 8x8, 4 octaves | 91.2 ms | 4186.3 ms | **45.9x faster** |
| snoise2 field, 32k points (serial path) | 1.1 ms | 18.4 ms | **16.7x faster** |
| snoise3 field, 4M points, coords +/-5000 | 66.1 ms | 1921.4 ms | **29.1x faster** |

Run-to-run variance is roughly ±10-25% on the Mojo column (it is a 72-thread
box shared with other jobs, so thread scheduling moves it) and a few percent on
the upstream column. Each row is the best of three timed passes. Across three
full runs `snoise2 field, 4M points` landed at 39-53 ms, `tiled 8x8` at
84-107 ms and `coords +/-5000` at 52-66 ms; treat those three as a band, not as
a point value. The optimisation pass below changed no kernel, so these are the
shipped numbers.

Read the numbers honestly:

- **The scalar row is a loss, and it is expected to be.** Reaching a compiled
  kernel one point at a time from Python means a ctypes marshalling cost of
  roughly a microsecond per call, against a C extension that Python calls
  directly. `mns.snoise2(x, y)` is the wrong way to use this library; it exists
  for API compatibility. `mns.snoise2(xs, ys)` is the right way.
- **The upstream column for the field rows is a Python `for` loop**, because
  upstream has no array API and that is the only way to evaluate a field with
  it. So the field speedups measure "a loop of 4M ctypes calls into a Mojo
  kernel" against "a loop of 4M calls into a C extension from Python". Most of
  that gap is the loop, not the kernel.
- The 32k row is below the 262144-point threading threshold and runs serial on
  one core, which is why it is the smallest field speedup rather than the
  largest.

### What the scalar row is actually made of

Measured on this box, per call, for the 8-argument `mns_noise2` signature:

| | ns/call | 200k calls |
| --- | ---: | ---: |
| `ctypes` marshalling of 8 float args (measured with a no-op C function) | 2263 | 453 ms |
| all four lookup tables rebuilt on the stack | ~70 | ~14 ms |
| the `noise2` kernel itself | ~90 | ~18 ms |
| ctypes call overhead for 2 args (baseline) | 913 | 183 ms |

So **75% of that row is `ctypes` marshalling** and about 3% is the kernel. The
scalar path is not slow because the kernel is slow; it is slow because eight
arguments have to cross a C ABI that Python reaches through `ctypes`. Nothing
on the Mojo side can move that number — it is fixed by the signature, which has
to keep upstream's argument list.

The arithmetic itself, measured in-process with no Python and no FFI, is
**26.7 ns/point for `snoise2` and 44.9 ns/point for `snoise3`** on one core.
Across the threading threshold that becomes 4.1 and 4.3 ns/point on up to 64
threads, which is where the field rows come from.

### Optimisation attempts, and why the code is unchanged

Every rung of the escalation order was measured on this box. None of them paid,
so `src/ported.mojo` is byte-for-byte what it was:

- **Hoisting the tiling test so the flat path skips `grad4`/`simplex`.** This
  removes 34 of the 46 vector stores from a scalar `mns_noise2` call and cuts
  its stack frame from 1480 to 40 bytes — it works, and it is measurable. But
  the only way to stop the compiler sinking those stores back is `@no_inline`
  on the table builders, and a function returning a `stack_allocation` pointer
  cannot be `@no_inline` on this toolchain: the caller's frame is not reserved,
  and every scalar result comes back wrong. The upside was ~70 ns of a
  2980 ns call (1.7%) against a correctness hazard. Reverted.
- **Caching the lookup tables.** Not possible: Mojo has no module-level
  variables (`global variables are not supported`), so there is nowhere to put
  a one-time-initialised table. The tables are rebuilt per call by necessity.
- **SIMD over the field loop.** Prototyped and measured: an 8-lane
  `SIMD[DType.float32, 8]` `noise2` reading its gradient from a 1024-byte
  precomputed LUT instead of the dependent `perm[perm[...]]` chain runs
  **9.9 ns/point against the scalar path's 26.9 ns/point — 2.7x faster**. It
  is not shipped because it is not the same kernel: 1.4% of points come back
  off by more than 1e-3 (worst 1.44), which is a different simplex cell, not
  float32 rounding. The gradient LUT itself was verified correct against
  upstream on every one of 524288 test points; the divergence is in how the
  vector lane selects corners, and closing it was not achieved. Correctness is
  not negotiable, so this stays out.
- **Threading.** `parallelize` is live via `from max.algorithm import
  parallelize` and is already used. Swept against the shipped thresholds:
  serial runs at ~30 ns/point and threaded at ~4 ns/point, and the crossover
  lands on the existing 262144-point boundary, so that number is right.
  Worker count swept at 8/16/32/64/128 over 262k, 1M, 4M and 16M points: 64
  wins at every size real callers pass, and 128 only wins at 16M points. Chunk
  size swept 1024..131072: 16384 is within noise of the best. Both constants
  stay.
- **GPU.** `max.gpu.host` is available and compiles on this toolchain — it is
  the live path, not a missing one. It is still the wrong tool here. These
  kernels are ~26.7 ns/point on one core for 12 bytes of traffic, about
  **0.45 flops per byte**, well under the ~2 flops/byte where a GPU starts to
  pay, and each point needs 3-4 dependent byte loads whose latency a GPU
  cannot hide behind the transfer. A 4M-point 2D field is 32 MB across HBM2 at
  ~1.8 TB/s, so the copies alone cost more than the entire CPU computation.
  No GPU path is shipped, and the reason is arithmetic intensity, not
  availability. (`nvidia-smi` reported 15215 MiB free, so nothing was skipped
  for want of memory.)

## How it works

**FFI.** `src/ported.mojo` compiles to `dist/libmojo-noise-simplex.so` with
`mojo build --emit shared-lib`, exporting six C-ABI symbols. Python loads it
with `ctypes`; there is no C extension module and no Python/C glue to compile.
`@export` rejects parametric functions and an inferred pointer origin, so every
buffer crosses as an `Int` address and is rebuilt inside the kernel as
`Pointer[T, AnyOrigin[mut=True]]`.

**API.** `simplex.py` mirrors upstream's names, argument order and defaults
exactly. Scalars take a `ctypes.c_float` path that narrows and calls directly;
arrays are broadcast, made C-contiguous float32, and handed to the `*_field`
kernel in one call. Field output is float32, matching the kernel's own width.

**Memory layout.** Coordinates in, values out, all flat float32, all
C-contiguous. One `ctypes` call per field, no per-element Python. The lookup
tables live in a stack buffer for the duration of one exported call — 944 bytes
total — built with 46 vector stores regardless of field size, never per point.

**Threading.** Field drivers run serially below 262144 points and above that
split the field into fixed 16384-point chunks across at most 64 threads. Fixed
chunks mean only the last one is ragged, and chunking is bit-identical to the
serial path — `tests/test_exports.py` asserts that one point either side of the
threshold produces the same bytes. All three constants were re-measured against
alternatives on this box and none beat them: the serial/threaded crossover lands
on 262144, worker counts of 8/16/32/64/128 favour 64 at every size real callers
pass (128 only wins above 4M), and chunk sizes from 1024 to 131072 are all
within noise of 16384.

**Parallelism note.** `parallelize` moved from `std.algorithm` to
`max.algorithm` on this toolchain; the import in `ported.mojo` is
`from max.algorithm import parallelize` and the `max` package is pinned in
lockstep with `mojo`.

**No GPU path, and that is a measured decision rather than a missing one.**
`from max.gpu.host import DeviceContext` and `from max.gpu import block_idx,
thread_idx` both resolve and compile on this toolchain, and `nvidia-smi`
reported 15215 MiB free, so nothing was skipped for want of an API or of
memory. The kernels just are not a GPU workload: `noise2` costs 26.7 ns/point on
one core for 12 bytes of traffic, about 0.45 flops per byte against the ~2
flops/byte where offload starts to pay, and each point needs 3-4 dependent
byte loads whose latency a GPU cannot hide behind the transfer. For a 4M-point
2D field the 32 MB of copies alone exceed the whole CPU computation. A CPU-only
port is the correct answer for this library.

## Testing

`pixi run test` runs 127 tests. Every exported kernel is named directly in
`tests/test_exports.py`, called through `ctypes`, and asserted on its return
value; that file also parses `src/*.mojo` and fails if a kernel is exported
without appearing in its coverage list. `tests/conftest.py` rebuilds the library
at configure time and aborts the run if `dist/libmojo-noise-simplex.so` is
absent or unloadable, so a test cannot pass without having exercised the
compiled kernel.

`tests/reference.py` is an independent NumPy transcription of upstream's C
source, used for the two things upstream cannot be asked about: `snoise4` with a
`lacunarity` argument, which **segfaults the installed `noise` 1.2.1** whenever
`lacunarity` is passed, and an aggressive `lacunarity=3.5`. The reference is
itself checked against upstream on every entry point, so it is not a second
unverified oracle.

Parity is asserted at three coordinate scales (|x| < 20, 500 and 5000), across
octave counts 1 to 12, both `persistence`/`lacunarity` regimes, and all three
tiling branches.

## Related

- [mojo-plotly](https://github.com/lee101/mojo-plotly) — data reduction kernels in Mojo
- [mojo-sklearn](https://github.com/lee101/mojo-sklearn) — scikit-learn's core estimators in Mojo

MIT.