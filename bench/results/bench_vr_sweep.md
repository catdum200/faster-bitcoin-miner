threads 1, 20 interleaved rounds of ~1.00 s per kernel; core GHz: scalar 3.21, ymm 3.29, zmm 2.66
| kernel | MH/s median | IQR | min..max | ns/hash/core | ~core cycles/hash | vs avx512 [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| cpuminer-opt16 | 27.243 | 26.572..27.572 | 25.187..27.978 | 36.71 | 97 | 0.967x [0.934, 0.979] |
| avx512 | 28.385 | 27.773..28.540 | 26.450..28.911 | 35.23 | 94 | 1.000x [1.000, 1.000] |
| avx512-vr@16 | 26.904 | 26.449..27.327 | 25.563..27.825 | 37.17 | 99 | 0.955x [0.944, 0.969] |
| avx512-vr@32 | 29.803 | 29.276..30.308 | 28.311..30.722 | 33.55 | 89 | 1.064x [1.057, 1.077] |
| avx512-vr@64 | 31.791 | 31.452..32.074 | 30.633..32.664 | 31.46 | 84 | 1.126x [1.110, 1.144] |
| avx512-vr@128 | 32.522 | 32.197..32.813 | 31.010..33.726 | 30.75 | 82 | 1.149x [1.137, 1.160] |
| avx512-vr@1024 | 32.223 | 31.817..32.967 | 30.732..33.654 | 31.03 | 82 | 1.151x [1.128, 1.166] |

ns/hash/core = threads / rate. ~core cycles/hash = ns/hash/core x measured core GHz
for the kernel's widest vectors (scalar, ymm or zmm license).
