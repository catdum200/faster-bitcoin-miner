# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A SHA-256d (Bitcoin proof-of-work) search engine for x86-64 CPUs, in C11
with intrinsics. It was built to answer "is there a more efficient way to
mine bitcoin?". The answer, with evidence, is in `README.md` and
`docs/RESULTS.md`:

- There is no algorithmic shortcut.
- The best software gain found is version rolling in SIMD lanes: 1.21x over
  cpuminer-opt, the best existing open-source CPU sha256d code.
- CPU/GPU mining is economically pointless: ~$0.002/year on this 4-core VM,
  ~$0.04/year estimated for an RX 9060 XT.

## Commands

```sh
make                 # builds ./fbm; prefers clang (see Conventions)
make check           # ./fbm test + tools/asmcheck.py hot-loop audit
./fbm test           # all correctness tests (~10 s); no per-test selector: the tests
                     # are the functions called from fbm_selftest() in src/selftest.c
make gen             # regenerate src/gen/*.h and stats.json from gen/gen_kernels.py
make asan; make tsan # tests under sanitizers (these targets use gcc, clean afterwards)
make BASELINES=1     # also fetch cpuminer-opt v26.1 (GPL-2, pinned, not vendored) and link
                     # its kernels as cpuminer-opt16 / cpuminer-opt8
./fbm list           # kernels + CPU support
./fbm bench --kernels avx512,avx512-vr@128 --threads 1 --rounds 20 --seconds 1
                     # interleaved rounds, IQR, bootstrap 95% CIs; name@N = N rolled
                     # versions per nonce; --baseline NAME (default cpuminer-opt16 if built)
./fbm freq           # measured core clock for scalar / ymm / zmm code
./fbm mine --header <160 hex> [--start N --count N --versions N --threads N]
python3 gen/gen_kernels.py --ablation   # exact per-technique vector-instruction counts
python3 tools/economics.py --mhs N --watts W [--usd-per-kwh P] [--rent-usd-per-hour 0]
bench/run_all.sh     # regenerates bench/results/* (the source of every number in
                     # docs/RESULTS.md); ~12 min, needs an otherwise idle machine
bench/cpuminer/build.sh  # separate harness for pooler/cpuminer's AVX2 assembly
```

## Architecture

**Kernel contract (`src/kernels.h`).** A scan function hashes the rectangle
versions `[r0, r0+nr)` × nonces `[n0, n0+nn)`. It reports a candidate when
`bswap(H7) <= job->t7`:

- Kernels only ever compute H7 of the final hash, exactly. H7 = IV7 + e
  after round 60 of the second hash, which is the early exit.
- `t7` is the top 32 bits of the target: 0 for any real network target,
  larger for easy targets.
- Tests use looser `t7` values through the *same* compare, so production and
  tests share one code path.
- Callers re-hash every candidate with the reference. `fbm mine` aborts when a
  candidate fails its own filter, because that can only be a kernel bug.

**Version rolling.** `version(r) = base ^ (r << 13)`, using BIP 320 bits
13-28, `r < 65536`. So `r = 0` is the header's own version.

**Kernels are generated, not hand-written.** `gen/gen_kernels.py` is a
partial evaluator over the SHA-256d dataflow. Each value has a kind:

| kind | changes |
|------|---------|
| CONST | never |
| JOB | per job |
| LANE | per job, but differs per SIMD lane |
| NONCE | per nonce, same in all lanes |
| VAR | per lane and per nonce |

There are two modes:

- **`nonce`:** lanes hold nonces; the midstate is JOB and the nonce word W3 is
  VAR.
- **`vr`:** lanes hold rolled versions; the midstate is LANE and W3 is NONCE.
  That makes the whole block-2 message schedule NONCE: scalar work, shared by
  every lane.

The generator folds constants, applies CSE, orders additions by estimated
readiness, and removes code that is dead after the H7 root. For each target
(scalar/avx2/avx512) and mode it emits `src/gen/g_<target>_<mode>.h`,
containing `_job`, `_lane` and `_nonce` (scalar precompute into `J[]`, `L[]`
and `N[]`) plus `_body` (vector code written with `V_*` macros).
`src/gen/stats.json` records the op counts that `tools/asmcheck.py` checks the
compiled loops against.

**Instantiation.** `src/kern_template.h` turns each generated pair into
`SCAN_NONCE` and `SCAN_VR` scan functions. The ISA files (`kernel_scalar.c`,
`kernel_avx2.c`, `kernel_avx512.c`, `kernel_avx512vl.c`) only define `LANES`,
`V` and the `V_*` ops, then include the generated headers and the template.
`kernel_avx512vl.c` reuses the avx512 generated code on 256-bit vectors.

**VR scan loop.**

- Per chunk of `VR_GROUPS` (8) vectors of versions, the LANE values are
  precomputed once.
- Per nonce, a `noinline` wrapper runs `_nonce` into `N[]` in memory, so the
  bodies use memory or embedded broadcasts rather than GPR broadcasts. All 8
  bodies share it.
- Sharing one schedule across fewer than ~64 versions is a net loss (see
  the sweep in `docs/RESULTS.md`). `fbm_kernel_best(versions)` therefore picks
  VR kernels only when `versions >= 64`.

**Registry and driver.**

- `src/kernels.c`: the registry, ordered slow to fast, with CPUID dispatch.
  Entries whose `desc` starts with `"baseline:"` are never auto-selected.
- `src/miner.c` (`fbm_run`): gives each thread a contiguous nonce slice (a
  multiple of 64 nonces) and pins it. Candidates beyond the buffer capacity
  are counted in `out->n > out->cap`, never silently dropped.

**Tests (`src/selftest.c`).** The oracle is the portable reference SHA-256 in
`src/sha256.c`, itself checked against the FIPS vectors and against OpenSSL.
Every supported kernel, the baselines included, must:

- report *exactly* the reference's candidate set on random rectangles,
  including odd sizes and the ends of the nonce and version spaces;
- rediscover the real (version, nonce) of 9 mainnet blocks. For the 7 with
  rolled versions, the base version is offset so the answer sits at `r = 5`;
- cover a range exactly once under 1, 4 and 7 threads.

## Conventions

- **Speed claims.** Take them from interleaved `fbm bench` runs (≥ 20 rounds,
  with CIs) against `cpuminer-opt16`, or from generator op counts. Beating
  the naive `ref` kernel alone is a strawman; this was critique #1.
- **Cycles.** The TSC always ticks at 2.8 GHz. Report ns/hash, or core cycles
  using `fbm freq`: zmm code runs at ~2.7 GHz, scalar/ymm at ~3.2-3.3 GHz.
- **Compiler.** clang is the default. gcc 13 spills much more in AVX2 (~20%
  slower) and puts `mov $imm; vpbroadcastd r32` pairs in the AVX-512 hot
  loops; `make check` flags both.
- **Generated code.** After editing `gen/gen_kernels.py`, run `make gen` and
  then `make check`. Never edit `src/gen/*.h` by hand.

## Progress (as of 2026-09-25)

**Process so far:** `docs/PLAN_v1.md`, then `docs/CRITIQUE.md` (an
independent adversarial review), then `docs/PLAN.md` (every finding mapped
to a change), then the implementation, then `docs/RESULTS.md`, whose numbers
come from `bench/results/`. The work is on branch
`claude/exciting-euler-zgd58e`.

**Measured on a 4-vCPU Cascade Lake VM** (AVX-512, no SHA-NI, no GPU):

- `avx512-vr`: 32.8 MH/s per core, 124.7 MH/s on 4 threads. That is 1.21x
  cpuminer-opt16 (95% CI 1.19-1.22) and 35x the naive `ref`.
- `avx512` (all prior-art tricks, no version rolling) only reaches parity
  (1.03x).
- ~84 core cycles/hash, against an ALU-port bound of 74.6.

**Fixed along the way:**

- worker threads silently dropped candidates beyond 4096;
- the generator now emits the next round's `h+K+W` before `Ch`, so
  `vpternlogd` can overwrite a dead register (~100 fewer copies per 16
  hashes);
- toggling `BASELINES` now rebuilds the kernel registry (a stamp file in
  `build/`).

**Open issues and next steps:**

- **256-bit kernels are front-end-bound.** `avx2` and `avx512vl` have 16-30 KB
  unrolled loops, more than the micro-op cache holds. A partially rolled body
  is the next step for AVX2-only CPUs.
- **`avx512vl-vr` is 3-5% slower than `avx512vl`** despite fewer
  instructions. The cause is unknown; this VM has no hardware counters.
- **Not implemented:**
  - a SHA-NI kernel (untestable here);
  - a Stratum/pool client (version rolling would also need BIP 310);
  - a GPU kernel. An RX 9060 XT is estimated at ~3 GH/s from published
    hashcat data; see the README.

## Environment gotchas

- **No counters.** The VM has no perf counters, RAPL or APERF/MPERF. Its CPU
  is shared, so benchmark only when nothing else, subagents included, is
  running.
- **ASan.** GCC 13's fake stack misaligns 64-byte locals (GCC PR 110027), so
  aligned AVX-512 stores fault. `make asan` therefore sets
  `ASAN_OPTIONS=detect_stack_use_after_return=0`.
- **cpuminer-opt under clang.** `bench/cpuminer-opt/fetch.sh` applies a
  one-line `sed` patch (the `mm512_perm128` function becomes a macro) so it
  compiles with clang. Its SHA code is untouched.
- **Network.** mempool.space and public GitHub clones work. Some benchmark
  sites return 403 to WebFetch; `curl` of OpenBenchmarking's
  `&export=csv` works.
