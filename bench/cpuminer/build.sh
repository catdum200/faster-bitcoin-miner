#!/bin/sh
# Build the cpuminer comparison harness from a checkout of pooler/cpuminer.
# Usage: bench/cpuminer/build.sh [path-to-cpuminer-checkout]
set -e
here=$(cd "$(dirname "$0")" && pwd)
src=${1:-$here/cpuminer-src}
if [ ! -f "$src/sha2.c" ]; then
    git clone --depth 1 https://github.com/pooler/cpuminer "$src"
fi
cc -O2 -I"$here/shim" -I"$src" -o "$here/cpuminer-bench" "$here/harness.c" "$src/sha2.c" \
    "$src/sha2-x64.S" -pthread
echo "built $here/cpuminer-bench (source: $src, commit $(git -C "$src" rev-parse --short HEAD))"
