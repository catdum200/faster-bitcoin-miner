#!/bin/sh
# Fetches cgminer 3.7.2's OpenCL SHA-256d kernels, the last maintained
# open-source GPU bitcoin kernels (2013; cgminer dropped GPU mining after 3.7),
# as prior-art baselines for tools/gpu_isacheck.py. Pinned to the v3.7.2 tag;
# not vendored (GPL-3 / public domain, see their headers).
set -eu
cd "$(dirname "$0")"
mkdir -p src
for f in poclbm130302.cl phatk121016.cl diakgcn121016.cl diablo130302.cl; do
    curl -sSfL -o "src/$f" "https://raw.githubusercontent.com/ckolivas/cgminer/v3.7.2/$f"
done
echo "fetched cgminer 3.7.2 kernels into bench/gpu-baselines/src/"
