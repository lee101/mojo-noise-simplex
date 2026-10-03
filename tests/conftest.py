"""Test setup.

Two things here matter for the honesty of the suite:

  * the compiled library is rebuilt from `src/ported.mojo` and asserted to be
    present before anything runs, so no test can pass without having loaded
    `dist/libmojo-noise-simplex.so`;
  * upstream `noise` is imported, not skipped. It is the oracle for nearly every
    numeric assertion, and a suite that quietly ran without it would be
    comparing the port against itself.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from mojo_noise_simplex import _lib  # noqa: E402


def pytest_configure(config):
    # Touch the library at configure time, not lazily inside the first test, so
    # a build failure is reported as one clear line rather than as a wall of
    # collection errors or, worse, as silently skipped tests.
    try:
        _lib.build(force=os.environ.get("MNS_FORCE_REBUILD") == "1")
    except _lib.SharedLibraryMissing as exc:
        raise pytest.UsageError(
            f"cannot build {_lib.LIB}: {exc}\n"
            f"run `pixi run build`, or set MOJO_NOISE_SIMPLEX_LIB to a "
            f"prebuilt shared library.") from exc


def pytest_sessionstart(session):
    path = _lib.build()
    if not os.path.exists(path):
        raise pytest.UsageError(
            f"compiled library missing: {path}. Run `pixi run build`.")
    try:
        _lib.lib()
    except _lib.SharedLibraryMissing as exc:
        raise pytest.UsageError(str(exc)) from exc


def pytest_report_header(config):
    path = _lib.build()
    size = os.path.getsize(path) if os.path.exists(path) else 0
    return [f"mojo-noise-simplex: {path} ({size} bytes)"]