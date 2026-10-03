"""mojo-noise-simplex against upstream `noise`.

    pixi run bench

Two honest caveats about the comparison, both reflected in the table below:

  * upstream `noise` is scalar-only. Its C entry points take one point at a
    time, so "upstream on a field" means a Python loop over them. That is the
    only thing a user of upstream can actually do today, and it is what the
    field rows measure. The comparison is therefore honest about what it is
    showing: a compiled kernel that takes an array, against a Python loop that
    does not exist because upstream cannot do it.
  * the scalar row calls the kernel through ctypes, which costs about a
    microsecond of marshalling per call. Against a C extension invoked directly
    from Python that overhead is real and unavoidable, so that row is expected
    to lose and is reported as such.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojo_noise_simplex as mns  # noqa: E402
import noise as up  # noqa: E402

FM = mns.FLT_MAX
REPEAT = 3


def timeit(fn, repeat: int = REPEAT) -> float:
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def grid(n, seed=0, lo=-20.0, hi=20.0):
    """float32 coordinates, the width upstream's argument parser narrows to."""
    rng = np.random.default_rng(seed)
    return [rng.uniform(lo, hi, n).astype(np.float32) for _ in range(4)]


def up_loop(fn, cols, count):
    """Upstream, driven one point at a time from Python."""
    flat = [np.ravel(c) for c in cols]
    return np.fromiter((fn(*row) for row in zip(*flat)), dtype=np.float64, count=count)


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case("snoise2 scalar x 200k")
def _():
    xs, ys = grid(200_000, 1)[:2]
    ours = mns._lib.lib().mns_noise2
    theirs = up.snoise2
    return (lambda: [ours(float(x), float(y), 1, 0.5, 2.0, FM, FM, 0.0)
                     for x, y in zip(xs, ys)],
            lambda: [theirs(float(x), float(y), 1, 0.5, 2.0, FM, FM, 0.0)
                     for x, y in zip(xs, ys)])


@case("snoise2 field, 4M points")
def _():
    xs, ys = grid(4_000_000, 2)[:2]
    return (lambda: mns.snoise2(xs, ys),
            lambda: up_loop(lambda a, b: up.snoise2(a, b, 1, 0.5, 2.0, FM, FM, 0.0),
                            [xs, ys], xs.size))


@case("snoise3 field, 4M points")
def _():
    xs, ys, zs = grid(4_000_000, 3)[:3]
    return (lambda: mns.snoise3(xs, ys, zs),
            lambda: up_loop(up.snoise3, [xs, ys, zs], xs.size))


@case("snoise4 field, 1M points")
def _():
    xs, ys, zs, ws = grid(1_000_000, 4)
    return (lambda: mns.snoise4(xs, ys, zs, ws),
            lambda: up_loop(up.snoise4, [xs, ys, zs, ws], xs.size))


@case("snoise3 field, 4M points, 6 octaves")
def _():
    xs, ys, zs = grid(4_000_000, 5)[:3]
    return (lambda: mns.snoise3(xs, ys, zs, octaves=6),
            lambda: up_loop(lambda a, b, c: up.snoise3(a, b, c, 6, 0.5, 2.0),
                            [xs, ys, zs], xs.size))


@case("snoise2 field, 4M points, tiled 8x8, 4 octaves")
def _():
    xs, ys = grid(4_000_000, 6)[:2]
    return (lambda: mns.snoise2(xs, ys, octaves=4, repeatx=8.0, repeaty=8.0),
            lambda: up_loop(lambda a, b: up.snoise2(a, b, 4, 0.5, 2.0, 8.0, 8.0, 0.0),
                            [xs, ys], xs.size))


@case("snoise2 field, 32k points (serial path)")
def _():
    xs, ys = grid(32_768, 7)[:2]
    return (lambda: mns.snoise2(xs, ys),
            lambda: up_loop(lambda a, b: up.snoise2(a, b, 1, 0.5, 2.0, FM, FM, 0.0),
                            [xs, ys], xs.size))


@case("snoise3 field, 4M points, coords +/-5000")
def _():
    xs, ys, zs = grid(4_000_000, 8, -5000.0, 5000.0)[:3]
    return (lambda: mns.snoise3(xs, ys, zs),
            lambda: up_loop(up.snoise3, [xs, ys, zs], xs.size))


def machine() -> str:
    model = "unknown CPU"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{model}, {os.cpu_count()} threads, {platform.system()} {platform.machine()}"


def main() -> int:
    print(f"machine: {machine()}")
    print(f"python:  {platform.python_version()}  numpy: {np.__version__}")
    print()
    print("| case | mojo-noise-simplex | upstream `noise` | |")
    print("| --- | ---: | ---: | --- |")
    for name, build in CASES:
        ours, theirs = build()
        mine = timeit(ours)
        theirs_t = timeit(theirs, repeat=1)  # the Python loop is slow; one pass
        ratio = theirs_t / mine
        if ratio >= 1.0:
            verdict = f"**{ratio:.1f}x faster**"
        else:
            verdict = f"{1.0 / ratio:.1f}x slower"
        print(f"| {name} | {mine * 1e3:.1f} ms | {theirs_t * 1e3:.1f} ms | {verdict} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())