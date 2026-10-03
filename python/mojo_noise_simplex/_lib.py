"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJO_NOISE_SIMPLEX_LIB") or os.path.join(
    ROOT, "dist", "libmojo-noise-simplex.so"
)

I = ctypes.c_int64
# The kernels compute in C `float`, exactly as upstream does, so they cross the
# ABI as c_float. Only the buffer addresses and the octave count are integers.
F = ctypes.c_float

# name -> (argtypes, restype)
_SIGNATURES = {
    "mns_noise2": ([F, F, I, F, F, F, F, F], F),
    "mns_noise3": ([F, F, F, I, F, F], F),
    "mns_noise4": ([F, F, F, F, I, F, F], F),
    "mns_noise2_field": ([I, I, I, I, I, F, F, F, F, F], None),
    "mns_noise3_field": ([I, I, I, I, I, I, F, F], None),
    "mns_noise4_field": ([I, I, I, I, I, I, I, F, F], None),
}


class BuildError(RuntimeError):
    pass


class SharedLibraryMissing(BuildError):
    """The compiled library is absent and could not be rebuilt.

    Raised instead of silently falling back to anything, so that a test which
    "passes" without ever loading `dist/*.so` is impossible.
    """


def mojo_command() -> list[str]:
    override = os.environ.get("MOJO_NOISE_SIMPLEX_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJO_NOISE_SIMPLEX_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/ported.mojo` into `dist/libmojo-noise-simplex.so` if stale."""
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib",
        os.path.join(SRC, "ported.mojo"), "-o", LIB,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except OSError as exc:
        raise SharedLibraryMissing(
            f"cannot run {' '.join(cmd)}: {exc}") from exc
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise SharedLibraryMissing((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib_path() -> str:
    """The shared library this package loads, without building anything.

    Use it to assert that a test really did exercise the compiled kernel.
    """
    return LIB


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        try:
            path = build()
        except (BuildError, OSError) as exc:
            raise SharedLibraryMissing(
                f"{LIB} could not be loaded or built: {exc}"
            ) from exc
        try:
            _lib = ctypes.CDLL(path)
        except OSError as exc:
            raise SharedLibraryMissing(f"cannot dlopen {path}: {exc}") from exc
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib




def main() -> int:
    """`python -m mojo_noise_simplex._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
