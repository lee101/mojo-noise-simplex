#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/dist"

# src/ported.mojo is the single compilation unit: build cost is fixed, so all
# kernels live together.
mojo build --emit shared-lib \
    "$repo_dir/src/ported.mojo" \
    -o "$repo_dir/dist/libmojo-noise-simplex.so"
