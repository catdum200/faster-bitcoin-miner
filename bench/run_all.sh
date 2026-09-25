#!/bin/sh
# Reproduce every measured number in docs/RESULTS.md (~12 minutes on a
# 4-core machine). Run on an otherwise idle machine.
set -e
cd "$(dirname "$0")/.."
out=bench/results
mkdir -p "$out"
make -s clean
make -s -j4 BASELINES=1
bench/cpuminer/build.sh >/dev/null
{
    date -u '+%Y-%m-%d %H:%M UTC'
    lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core|L2|L3)'
    grep -o -w -E 'avx2|avx512f|avx512vl|avx512bw|sha_ni' /proc/cpuinfo | sort -u | tr '\n' ' '
    echo
    ${CC:-clang} --version | head -1
    uname -r
} > "$out/env.txt"
./fbm freq > "$out/freq.txt"
./fbm bench --kernels all --threads 1 --rounds 20 --seconds 1 > "$out/bench_1thread.md"
./fbm bench --kernels ref,avx2,avx2-vr,cpuminer-opt8,avx512vl-vr,cpuminer-opt16,avx512,avx512-vr \
    --threads 4 --rounds 20 --seconds 1 > "$out/bench_4threads.md"
./fbm bench --kernels cpuminer-opt16,avx512,avx512-vr@16,avx512-vr@32,avx512-vr@64,avx512-vr@128,avx512-vr@1024 \
    --baseline avx512 --threads 1 --rounds 20 --seconds 1 > "$out/bench_vr_sweep.md"
{
    bench/cpuminer/cpuminer-bench 1 $((1 << 25)) 9
    bench/cpuminer/cpuminer-bench 4 $((1 << 26)) 9
} > "$out/pooler_cpuminer.txt"
python3 tools/asmcheck.py build > "$out/asmcheck.txt"
echo "results in $out"
