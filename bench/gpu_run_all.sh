#!/bin/sh
# Run this on the machine with the GPU (e.g. an RX 9060 XT). It measures
# everything the VM that wrote this code could not, into one file under
# bench/results/:
#   1. device and compiled-kernel info (wave size, spills);
#   2. the correctness tests on the real GPU;
#   3. nonce-lane vs version-lane hash rate, interleaved, with 95% CIs;
#   4. a sweep of versions per nonce and of work-group size;
#   5. the ISA audit of the binary the installed driver actually compiled
#      (needs llvm-objdump; skipped otherwise).
# Usage: bench/gpu_run_all.sh [fbm-gpu options, e.g. --platform 0 --device 0]
# Close games and video first: the GPU must be otherwise idle. Takes ~10 min.
set -u
cd "$(dirname "$0")/.."
[ -x ./fbm-gpu ] || make gpu || exit 1
mkdir -p bench/results
name=$(./fbm-gpu info "$@" | sed -n 's/^device: //p' | head -1 | tr -c 'A-Za-z0-9\n' '_')
out="bench/results/gpu-${name:-unknown}.txt"
{
    echo "## fbm-gpu info"; ./fbm-gpu info "$@"
    echo; echo "## fbm-gpu test"; ./fbm-gpu test "$@"
    echo; echo "## nonce vs vr, 20 rounds"
    ./fbm-gpu bench --rounds 20 --seconds 2 "$@"
    for v in 64 256 1024 16384; do
        echo; echo "## vr with $v versions per nonce"
        ./fbm-gpu bench --kernels nonce,vr --versions "$v" --rounds 6 --seconds 2 "$@"
    done
    for wg in 32 128 256; do
        echo; echo "## work-group $wg"
        ./fbm-gpu bench --rounds 6 --seconds 2 --wg "$wg" "$@"
    done
    echo; echo "## driver-compiled ISA"
    ./fbm-gpu dump --out bench/results/gpu-kernels.bin "$@" &&
        python3 tools/gpu_isacheck.py --binary bench/results/gpu-kernels.bin
} 2>&1 | tee "$out"
echo "wrote $out"
