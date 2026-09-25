# Results

Every number here was measured on one machine with `bench/run_all.sh`. The raw
outputs are in [`bench/results/`](../bench/results/).

## Setup

- **Machine:** a 4-vCPU KVM guest on an Intel Xeon (Cascade Lake, family 6 model
  85 stepping 7), nominal 2.8 GHz. It has AVX2 and AVX-512 F/BW/DQ/VL/VNNI and
  **no SHA-NI**. There is no GPU, and the VM has no power or cycle counters.
- **Measured core clock** (`fbm freq`, dependent-add chains): 3.29 GHz scalar, 3.30 GHz 256-bit, **2.76 GHz 512-bit** (AVX-512 frequency license). The TSC
  always ticks at 2.8 GHz, so "cycles" below are **core** cycles, computed as
  ns/hash × the clock measured for the kernel's widest vectors. That corrects
  the plan's TSC error (critique #5).
- **Method** (critique #6):
  - Each benchmark runs **20 interleaved rounds**. Each round runs every
    kernel once for about 1 s, and the order rotates between rounds.
  - Threads are pinned, and calibration runs double as warm-up.
  - Every run uses the same pseudo-random header and the production filter
    (t7 = 0, as for any real target).
  - Speedups are medians of per-round ratios, with percentile-bootstrap 95%
    confidence intervals (2000 resamples).
- **Baselines** (critique #1):
  - **cpuminer-opt v26.1** (JayDDee), the fastest open-source CPU sha256d code
    we know. Its 16-way AVX-512 and 8-way kernels are linked in-process
    (`make BASELINES=1`), built as its own build does (`-O3 -march=native`)
    with clang, its fastest compiler here. They run under the same harness
    and pass the same correctness tests as fbm's kernels.
  - **pooler/cpuminer 2.5.1** (commit 5f02105), hand-written AVX2 assembly. It runs in its own
    harness, `bench/cpuminer`, so it is *not* interleaved with the others.
  - The naive reference, a portable C SHA-256 over the whole header twice per
    nonce, serves as the oracle.
- **Build:** clang 18 `-O2`. The SIMD kernels are selected at run time.

## Headline

Hash rate with the production filter, median of 20 interleaved rounds.
Speedups are against cpuminer-opt's 16-way AVX-512 kernel, with 95% CIs.
Core cycles/hash are from the 1-thread run.

| kernel | what it is | 1 thread MH/s | vs cpuminer-opt16 | 4 threads MH/s | vs cpuminer-opt16 | core cycles/hash |
|---|---|---:|---:|---:|---:|---:|
| `ref` | naive: portable C, whole header hashed twice per nonce (the oracle) | 0.93 | 0.034x | 3.52 | 0.034x | 3530 |
| `openssl` | naive, with OpenSSL 3 `SHA256()` | 0.60 | 0.022x | | | 5261 |
| `scalar` | midstate + folding + early exit, scalar C | 1.90 | 0.070x | | | 1731 |
| `scalar-vr` | ... + version rolling | 2.15 | 0.079x | | | 1528 |
| pooler/cpuminer | prior art: hand-written AVX2 assembly (separate harness, not interleaved) | 11.38 | | 43.88 | | |
| `avx2` | 8 nonces per AVX2 vector | 11.02 | 0.406x | 41.45 | 0.402x | 288 |
| `avx2-vr` | 8 rolled versions per AVX2 vector | 13.03 | 0.476x | 49.80 | 0.479x | 243 |
| `cpuminer-opt8` | prior art: 8-way, AVX-512VL build | 19.28 | 0.705x | 73.15 | 0.701x | 164 |
| `avx512vl` | 8 nonces per 256-bit AVX-512VL vector | 20.59 | 0.753x | | | 154 |
| `avx512vl-vr` | 8 rolled versions per 256-bit AVX-512VL vector | 19.89 | 0.733x | 76.47 | 0.737x | 159 |
| `cpuminer-opt16` | **prior art: 16-way AVX-512, the best existing CPU code** | 27.41 | 1 | 103.71 | 1 | 101 |
| `avx512` | 16 nonces per AVX-512 vector (all prior-art tricks) | 27.95 | 1.027x [1.015, 1.043] | 107.67 | 1.039x [1.029, 1.049] | 99 |
| **`avx512-vr`** | **16 rolled versions per AVX-512 vector (+ version rolling)** | **32.82** | **1.208x [1.184, 1.227]** | **124.70** | **1.207x [1.194, 1.223]** | **84** |

What this says:

- **Version rolling is the only real gain over the state of the art: +21%
  on 1 and 4 threads.** Everything else (midstate, folding, early exit,
  16-way AVX-512) reproduces prior art: `avx512` is at parity with
  cpuminer-opt (+3-4%).
- **Against a naive miner it is 35x** (32.82 vs 0.93 MH/s per core; 124.7 vs
  3.5 MH/s on 4 cores).
- **The kernels run close to the hardware limit.** On this core, 512-bit
  rotates and shifts run only on port 0, and adds and ternary logic on ports
  0 and 5. The ALU-port bound is therefore (vector instructions) / 2 ports /
  16 lanes:
  - `avx512-vr`: 2387 / 32 = 74.6 cycles/hash, measured 84, which is **89%
    of the bound**;
  - `avx512`: 2847 / 32 = 89.0, measured 99, which is 90%.
- **512-bit beats 256-bit despite the clock drop** (2.76 vs 3.30 GHz). Each
  instruction does twice the work, and that matters because these unrolled
  loops are limited by instruction fetch for 256-bit code (see "Open
  issues").
- **Anomaly:** version rolling helps the zmm (+17%) and AVX2 (+18%) kernels,
  but not the 256-bit AVX-512VL one (−3%). See "Open issues".

## What each technique is worth (exact instruction counts)

Timings on a shared VM are noisy (critique #6), so the per-technique
breakdown uses exact counts instead. `python3 gen/gen_kernels.py --ablation`
builds the same computation with techniques switched off. It counts the
vector instructions per hash that survive constant folding and dead-code
elimination:

**AVX-512 (16 lanes)**, vector instructions per hash:

| step | per hash | vs previous | total reduction |
|---|---:|---:|---:|
| naive: 3 compressions per nonce, nothing precomputed | 298.5 | - | - |
| + midstate (T1): 2 compressions per nonce | 197.7 | -33.8% | 1.51x |
| + constant folding / precompute (T2, T3) | 185.4 | -6.2% | 1.61x |
| + early exit after round 60 of hash 2 (T4) | 177.9 | -4.0% | 1.68x |
| + version rolling, 128 versions per nonce (T7) | 149.2 (+4.2 scalar per hash for the shared schedule) | -16.2% | 2.00x |

**AVX2 (8 lanes)**, vector instructions per hash:

| step | per hash | vs previous | total reduction |
|---|---:|---:|---:|
| naive: 3 compressions per nonce, nothing precomputed | 1196.2 | - | - |
| + midstate (T1): 2 compressions per nonce | 793.0 | -33.7% | 1.51x |
| + constant folding / precompute (T2, T3) | 748.5 | -5.6% | 1.60x |
| + early exit after round 60 of hash 2 (T4) | 718.6 | -4.0% | 1.66x |
| + version rolling, 128 versions per nonce (T7) | 613.6 (+4.2 scalar per hash for the shared schedule) | -14.6% | 1.95x |

The counts predict the measurements:

- **T7 (version rolling).** It removes 16.2% of AVX-512 instructions, a
  predicted 1.19x speedup. Measured `avx512-vr` / `avx512` at 128 versions
  per nonce: 1.17x (1 thread) and 1.16x (4 threads) in the headline runs,
  and 1.15x in the sweep below.
- **T1-T4.** The step from naive to early exit is 1.68x fewer
  instructions. But measured, these tricks only bring `avx512` level with
  cpuminer-opt, which already implements all of them. That is critique #1's
  point.

## How many versions must share one schedule? (critique #4)

In the version-lane layout, each nonce's block-2 message schedule is
computed once in scalar code, and each vector of 16 versions then skips ~460
vector instructions. The critique warned that the scalar work is not free:
scalar rotates compete with vector rotates for port 0. The sweep varies how
many versions share each schedule, `avx512-vr@N` (1 thread, 20 interleaved
rounds, relative to the nonce-lane `avx512` kernel):

| kernel | MH/s median | IQR | min..max | ns/hash/core | ~core cycles/hash | vs avx512 [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| cpuminer-opt16 | 27.243 | 26.572..27.572 | 25.187..27.978 | 36.71 | 97 | 0.967x [0.934, 0.979] |
| avx512 | 28.385 | 27.773..28.540 | 26.450..28.911 | 35.23 | 94 | 1.000x [1.000, 1.000] |
| avx512-vr@16 | 26.904 | 26.449..27.327 | 25.563..27.825 | 37.17 | 99 | 0.955x [0.944, 0.969] |
| avx512-vr@32 | 29.803 | 29.276..30.308 | 28.311..30.722 | 33.55 | 89 | 1.064x [1.057, 1.077] |
| avx512-vr@64 | 31.791 | 31.452..32.074 | 30.633..32.664 | 31.46 | 84 | 1.126x [1.110, 1.144] |
| avx512-vr@128 | 32.522 | 32.197..32.813 | 31.010..33.726 | 30.75 | 82 | 1.149x [1.137, 1.160] |
| avx512-vr@1024 | 32.223 | 31.817..32.967 | 30.732..33.654 | 31.03 | 82 | 1.151x [1.128, 1.166] |

**The critique was right about the cost.** With one vector per schedule (16
versions), the scalar schedule costs more than it saves: 0.955x. From 32
versions it pays, and it saturates at 128. The miner uses 128 per nonce, and
BIP 320 allows 65,536, so the scalar design stays. The proposed zmm
schedule-table redesign would only have helped below 64 versions. `fbm mine`
picks the version-lane kernels only when rolling ≥ 64 versions.

## Compiled hot loops (critique #2)

`tools/asmcheck.py` runs as part of `make check`. It finds each kernel's hot
loop in the object code, including the body when clang outlines it, and
counts what runs per iteration:

```
hot loop                insns    alu  expected gpr-bcast  reg-mov  spill
fbm_scan_avx2            6106   5753      5749         0       32    147
fbm_scan_avx2_vr         5158   4911      4909         0       20     62
fbm_scan_avx512vl        3208   2849      2847         1      166    119
fbm_scan_avx512vl_vr     2618   2387      2387         0      155     45
fbm_scan_avx512          3207   2849      2847         1      166    119
fbm_scan_avx512_vr       2617   2387      2387         0      155     45
ok
```

- **The ALU count matches the generator's count** within the 2-4
  instructions of the final compare, so the compiler adds no work.
- **There are no register broadcasts** (the 1 is the per-iteration nonce
  base).
- **The same audit fails on a gcc 13 build.** gcc puts **82 and 73**
  `vpbroadcastd r32` in the AVX-512 loops, each an extra port-5 micro-op.
  It also has **717 and 647** spill stores in the AVX2 loops, against 147
  and 62 for clang. That is why clang is the default (AVX2: 11.0 vs about
  9.0-9.5 MH/s).
- **Register copies.** `vpternlogd` overwrites one of its inputs, so each
  Ch and Maj on live state needs a copy first, and zmm copies are only
  partly eliminated at rename. A microbenchmark showed 4 copies per 8 adds
  costing 15-23%. The generator now emits the next round's `h + K + W`
  before `Ch`, which makes `Ch` the last use of `g` and lets clang overwrite
  it. That cut copies from **274 to 166** (nonce layout) and **241 to 155**
  (version layout) per 16 hashes. Maj's copy cannot be avoided without an
  extra add.

## Correctness

`make check` passes, and so do `make asan` (ASan+UBSan) and `make tsan`
(ThreadSanitizer). The tests:

- **Oracle checks.** The reference SHA-256 must match the FIPS 180 vectors,
  including the 1M-'a' streaming vector, and must match OpenSSL for every
  length from 0 to 299.
- **Candidate-set equality.** Every kernel, the cpuminer-opt wrappers
  included, must report **exactly** the reference's candidate set in six
  cases:
  - loose and tight thresholds;
  - almost-every-lane-hits;
  - odd sizes;
  - the end of the nonce and version spaces;
  - the production threshold t7 = 0.
- **Real blocks.** Every kernel must rediscover the real (version, nonce) of
  9 mainnet blocks: genesis, 125552, and 7 blocks with ASIC-rolled versions
  (800000, 850000, 900000, 910000, 968555, 968562, 968566). For those 7 the
  job starts from version ⊕ (5 << 13), so the kernel has to find the
  solution by rolling.
- **Thread partitioning.** `fbm_run` with 1, 4 and 7 threads must cover a
  range exactly once. This test found a real bug: worker threads silently
  dropped candidates beyond 4096. Real targets never produce that many, but
  easy targets (regtest, low-difficulty shares) would have lost solutions.
  Fixed; overflow is now counted and reported.
- **Mining itself.** `fbm mine` re-hashes every candidate. A candidate that
  fails its own 32-bit filter is a kernel bug, so the run aborts instead of
  discarding it (critique #3).
- **Sanitizer caveat.** Under GCC 13, ASan's fake stack misaligns 64-byte
  locals (GCC PR 110027), which faults on aligned AVX-512 stores. The `asan`
  target disables only `detect_stack_use_after_return`. The crash is in the
  toolchain, not this code: it disappears with either sanitizer alone and
  with GCC's fix.

## Economics: is any of this worth it? (critique #7)

`python3 tools/economics.py --mhs 124.7 --watts 30` uses live mempool.space
data. The best 4-thread rate is 124.7 MH/s. The VM has no power counters, so
30 W is an estimate: ~5-10 W per busy Cascade Lake core at package level.

Network (live mempool.space): height 968570, hashrate 9.26e+20 H/s, difficulty 1.328e+14, BTC $83731, reward 3.1515 BTC/block (subsidy 3.125 + fees 0.0265)

| quantity                               | value |
|----------------------------------------|---|
| This miner                             | 124.7 MH/s |
| Share of network hashrate              | 1.3e-13 |
| Expected revenue                       | 5.95e-11 BTC/day = $0.0018 per year |
| Expected time to find a block solo     | 1.45e+08 years |
| Chance of a block within a year        | 6.9e-09 |
| Energy efficiency (at 30 W, estimated) | 2.41e+05 J/TH |
| Antminer S21 XP (spec)                 | 13.5 J/TH, 270 TH/s |
| CPU energy per hash vs ASIC            | 1.8e+04x worse |
| Electricity at $0.10/kWh               | $26.28 per year |
| Revenue / electricity cost             | 6.9e-05 |
| Cloud VM rent at $0.15/h               | $1314 per year (7.2e+05x revenue) |
| ASIC revenue for comparison            | 0.000129 BTC/day = $3941 per year per machine |
| ASIC electricity at $0.10/kWh          | $3193 per year per machine |

**Verdict: CPU mining bitcoin is a pure loss, and a 1.2x software gain
cannot change that.** The machine would earn about **$0.002 per year**, while
its electricity costs about $26 and renting it costs about $1,300. Solo, it
would find a block about once every **145 million years**. Per hash it uses
**~18,000x more energy** than a current ASIC, a gap no software can close.
The project's value is engineering and education, not income. Cloud
providers' terms often forbid mining anyway.

## So what *does* make bitcoin mining more efficient?

In order of impact:

1. **Hardware.** ASICs do ~13.5 J/TH (Antminer S21 XP spec). This CPU does
   ~2.4×10^5 J/TH. That is four orders of magnitude, against 1.2x for the
   best software trick found here.
2. **Electricity price.** Even that ASIC barely beats its power bill at
   $0.10/kWh: about $3,900 per year of revenue against about $3,200 of
   electricity, before hardware, cooling and hosting. Profitable mining
   today is decided by power cost, roughly $0.05/kWh or below, or by reusing
   the waste heat.
3. **Version rolling (BIP 320 / overt AsicBoost).** This is the one
   algorithmic efficiency gain on SHA-256d that matters. ASIC firmware
   already does it: 14 of the 15 most recent blocks we checked have rolled
   version bits.
4. **Pools** don't make hashing more efficient, but they turn a
   145-million-year lottery into a steady (tiny) income.

For an individual with a CPU, the most efficient way to "mine" bitcoin is
not to mine: the electricity and hardware money buys more bitcoin directly.

## Open issues

- **Version rolling does not help the 256-bit AVX-512VL kernel.**
  `avx512vl-vr` measures 0.93-0.97x of `avx512vl` at any version count
  (`fbm bench --kernels avx512vl,avx512vl-vr@64,avx512vl-vr@128,avx512vl-vr@1024`).
  - **Why it is not ALU-bound.** Every hot loop is **16-30 KB of code**
    (`avx512vl` 20.3 KB, `avx512vl-vr` 16.4 KB, `avx512-vr` 17.8 KB,
    `avx2` 29.9 KB). That is far more than Cascade Lake's micro-op cache,
    so the loops run from the legacy decoders at ≤ 16 bytes/cycle.
  - **The front end is the limit for 256-bit code.** For `avx512vl` that
    bound is 20,334 B / 16 = 1271 cycles per 8 hashes, and it measures
    ~1230. Its 3-port ALU bound (950) is not the limit.
  - **zmm is different.** It needs only 2 ALU ports' worth of work per
    cycle, stays ALU-bound, and so gains from doing less work.
  - **Unexplained remainder.** `avx512vl-vr` has 20% fewer bytes and the
    same memory-operand mix as `avx512-vr`, yet runs 3-5% *slower* than
    `avx512vl`. Without hardware counters, which this VM does not expose,
    we could not find why.
  - **Practical effect: none here.** `fbm mine` uses the zmm kernels on
    AVX-512 CPUs.
- **Next improvement for AVX2-only CPUs.** Because the unrolled 256-bit
  kernels are front-end-bound, a partially rolled body that fits the
  micro-op cache (for example, a loop over 8-round blocks) is the most
  promising next step. pooler/cpuminer's rolled assembly already matches our
  unrolled `avx2` kernel (11.4 vs 11.0 MH/s). Only `avx2-vr` beats it
  (13.0, +14%).
- **SHA-NI.** Most current CPUs (AMD Zen, Intel Ice Lake and later) have
  SHA extensions. A SHA-NI kernel would likely beat AVX2 there, but it could
  not be tested on this CPU, so none is shipped.
- **One VM.** These numbers are from one VM. Another CPU, especially a
  non-Intel or non-AVX-512 one, will rank the kernels differently.

## Where the critique was right, and where it was not

| critique finding | outcome |
|---|---|
| #1 the baseline was a strawman, and T1-T5 are prior art | **Right.** Midstate, prehash, folding, early exit and 16-way AVX-512 together only reach parity with cpuminer-opt (row `avx512`). The one idea that beats it is version rolling (T7) |
| #2 codegen matters (gcc GPR broadcasts) | **Right.** gcc 13 puts 82 and 73 `vpbroadcastd r32` in the AVX-512 hot loops and ~700 spill stores in AVX2. clang has 0-1 and ~150. `tools/asmcheck.py` now guards this in `make check` |
| #3 silent-drop risks | The compare was already shared by tests and production (t7-based, valid for easy targets). The new tests then found the 4096-candidate cap bug above |
| #4 the scalar schedule is not free | **Right when it is shared by only 16 versions.** In the sweep, `@16` is 17% slower than `@128` and 4.5% slower than no version rolling at all. **Fine when shared by ≥ 64.** The design shares it across 128, so the proposed vector-table redesign was not needed |
| #5 TSC ≠ core clock | **Right:** 2.7-2.8 GHz under zmm vs 3.2-3.3 GHz for scalar/ymm |
| #6 statistics | Adopted: interleaved rounds, IQR, bootstrap CIs |
| #7 economics | Adopted, with live data (see "Economics") |
| #8 dead `a` work in rounds 57-60 | Already removed by the generator's dead-code elimination (Σ0 appears 116 times = 60 + 56). The phatk constant trick is worth 1 add per 16 hashes and was skipped |
| #9 two-stream interleave | Dropped. One stream already runs at ~90% of the ALU-port bound |

## Reproduce

```sh
make clean && make BASELINES=1 && make check   # build (fetches cpuminer-opt), test, audit
bench/run_all.sh                               # every table above, into bench/results/
python3 gen/gen_kernels.py --ablation          # exact per-technique instruction counts
python3 tools/economics.py --mhs 124.7 --watts 30
```
