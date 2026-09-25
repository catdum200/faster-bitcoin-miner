# Plan v1 — a more efficient Bitcoin (SHA-256d) miner

Status: first draft, written before any code. It will be critiqued by an
independent agent and then revised (see `CRITIQUE.md` and `PLAN.md`).

## 0. Honest framing

* Bitcoin proof-of-work is `SHA256(SHA256(header80))` compared against a target.
  There is **no known shortcut**: ideas such as predicting nonces or using ML
  to guess good nonces cannot work unless SHA-256 is broken. So "more
  efficient" can only mean one of these:
  1. fewer operations per hash attempt (algorithmic engineering that exploits
     the *structure* of the header, not a weakness of SHA-256),
  2. more attempts per second from the same silicon (parallelism, SIMD),
  3. better hardware or cheaper energy (ASICs). This is out of scope for code,
     but it dominates real-world economics, and we will quantify it.
* Hardware available: 4-core Intel Xeon (Cascade Lake, model 85) VM at 2.8 GHz,
  with AVX2 and AVX-512 F/BW/DQ/VL/VNNI. It has **no SHA-NI** and no GPU.
* Goal: build the most efficient SHA-256d search engine we can for this
  hardware, measure the contribution of every trick separately, and give an
  honest economic verdict.

## 1. Deliverables

* A C99/C11 project (gcc 13, Makefile) that builds one binary, `fbm`:
  * `fbm test`: correctness self-tests.
  * `fbm bench`: benchmark every kernel, single-threaded and multi-threaded.
  * `fbm mine <80-byte-header-hex> [--threads N] [--start S] [--count C]`:
    search a nonce range and print the solutions.
* `docs/RESULTS.md`: measured numbers, a per-technique breakdown, and the
  economics.

## 2. Techniques (each one benchmarked on its own)

| id | technique | expected effect |
|----|-----------|-----------------|
| T0 | Naive baselines: byte-oriented reference SHA-256 over the full 80-byte header, run twice (3 compressions per nonce); OpenSSL `SHA256()`; Python `hashlib` | reference points |
| T1 | **Midstate**: the first 64-byte block (version, prev hash, 28 bytes of merkle root) is constant per job, so hash it once. That leaves 2 compressions per nonce instead of 3 | ~1.5x |
| T2 | **Constant folding in block 2 of hash 1**: W0..W2 (merkle tail, time, bits) are constant, so rounds 0-2 are nonce-independent (precompute the state) and round 3 is a single add. W4..W15 are padding constants (fold K+W). W16 and W17 are constant. W18..W32 have large constant sub-terms | ~5-10% |
| T3 | **Constant folding in hash 2**: the IV is constant and W8..W15 are padding constants, so round 0 is mostly precomputable and K+W folds. The compiler can do this itself because everything is a compile-time constant | small |
| T4 | **Early exit**: difficulty >= 1 needs the top 32 bits of the LE hash to be 0, i.e. final `H7 == 0`. `H7 = IV7 + e` from round 60, so we stop after round 60 (skip rounds 61-63 and W61..W63, and compute only `e` in round 60, not `a`). Test `e == 0xa41f32e7`, then fully verify candidates with the reference | ~5% |
| T5 | **SIMD**: AVX2 8-way and AVX-512 16-way (one nonce per 32-bit lane). AVX-512 has native rotate (`vprord`) and `vpternlogd` (Ch, Maj, and 3-way XOR in 1 instruction each), so it needs far fewer instructions per round | 8-16x over scalar |
| T6 | **Multithreading**: pthreads, one thread per core, disjoint nonce ranges | ~4x |
| T7 | **Version rolling (overt ASICBoost, BIP 320)**: block 2 of hash 1 depends only on (merkle tail, time, bits, nonce), not on the version. Rolling version bits 13-28 gives many midstates that share *identical* block-2 message schedules. Layout: 16 lanes = 16 versions for the **same** nonce. The block-2 schedule is then uniform across lanes, so it is computed in **scalar** code (on otherwise idle scalar ports p1/p6) and consumed through broadcast memory operands. That removes ~15% of vector work | ~10-20% |
| T8 | Micro-optimizations: full unrolling; interleaving 2 independent vectors for ILP; checking the asm for spills and `vpbroadcastd` from GPR | measure |

## 3. Correctness strategy

* Reference SHA-256 checked against the FIPS 180-2 vectors ("", "abc", and the
  448-bit message).
* Real mainnet blocks, genesis (nonce 2083236893) and block 125552 (nonce
  2504433986): the double hash must meet the target, and every kernel must find
  the known nonce when scanning a range that contains it.
* Differential testing: random headers, every kernel against the reference
  over >= 2^20 nonces. The kernels take a **test mask** (a weaker
  `(H7 & mask) == 0` filter; production uses mask = 0xffffffff), so there are
  enough hits to compare exact nonce sets.
* The version-rolling kernel is tested the same way, with versions made
  explicit.
* Build and run the tests under `-fsanitize=address,undefined`.

## 4. Benchmark methodology

* Fixed work per run (for example 2^26 nonces), 5 repeats, report the median.
  Measure single-threaded and 4 threads.
* Report MH/s, ns/hash, and cycles/hash at TSC frequency.
* Record the environment (CPU, flags, compiler, compile flags).
* Caveats: this is a VM (noisy neighbours), AVX-512 frequency licenses apply,
  and there is no RAPL, so energy cannot be measured directly.

## 5. Economics

* Take the current network hashrate, block subsidy (3.125 BTC), fees and price,
  and work out the expected BTC per day and the solo time-to-block for the
  measured hashrate. Compare with a modern ASIC in J/TH, using clearly
  labelled TDP-based estimates for the CPU.

## 6. Milestones

1. Reference SHA-256, header utilities, target math, and tests.
2. Naive baselines and the bench harness.
3. Optimized scalar kernel (T1-T4).
4. AVX2 kernel.
5. AVX-512 kernel.
6. Version rolling (T7).
7. Threads and the CLI.
8. Results and docs, then commit and push.

## 7. Risks

* gcc may not turn the rotate idiom into `vprord`, so use intrinsics
  (`_mm512_ror_epi32`).
* AVX-512 frequency throttling may eat part of the gain.
* VM noise, which the medians and repeats should absorb.
* An early-exit bug could silently drop valid solutions. The differential
  tests with a weaker mask guard against this.

## 8. Non-goals

* Stratum/pool networking. It adds no efficiency, and a CPU would earn about
  $0.
* GPU/FPGA/ASIC work, because this hardware is not available.
* "Quantum" or "ML nonce prediction", which do not work (see §0).
