# Plan v2: revised after the critique

History: `PLAN_v1.md` was the first draft. `CRITIQUE.md` is an independent
agent's adversarial review of it, with its own measurements on this VM. This
file is the improved plan that the code now follows. Core implementation (the
kernel generator) started while the review was running, so a few findings
were already handled by design; the table says which.

## 1. What the critique changed

| # | Finding | Response |
|---|---------|----------|
| 1 | **[critical]** The baseline was a strawman. Midstate, prehash, pre-extension, zero-word folding and early exit (T1–T5) are 2011–2016 prior art (pooler/cpuminer, phatk, cpuminer-opt). cpuminer-opt's 16-way AVX-512 kernel already reaches ~27 MH/s/core here | **Accepted.** The headline baseline is now **cpuminer-opt's 16-way AVX-512 kernel, built with clang, on this VM**. pooler/cpuminer's AVX2 assembly is a second baseline. The naive reference stays only as the correctness oracle and as a "what a naive miner does" data point. T1–T5 are credited as prior art. The only novel claim left is T7 (version-rolling schedule sharing, which cpuminer-opt lacks), with an honest ceiling of **≤ 1.18x** over prior art |
| 2 | **[critical]** Codegen matters as much as T7. gcc emits `mov $K; vpbroadcastd r32` (an extra p5 uop); clang uses `{1to16}` memory operands | **Accepted, already confirmed independently:** clang AVX2 is +20%. The default compiler is now clang (gcc is still supported). Add `tools/asmcheck.py`: count GPR broadcasts and spills inside each hot loop, and fail `make check` if the clang build's hot loop contains GPR broadcasts beyond a small budget |
| 3 | **[critical]** Gaps could drop solutions silently: the test mask differed from the production compare; H7==0 is invalid for easy targets; W3 vs nonce ranges; the reference re-check swallowed kernel bugs | **Mostly handled by design:** one compare in every mode, `bswap(H7) <= t7` with t7 = top 32 bits of the target. This is exact for real targets (t7 = 0) and correct for easy or regtest targets. Ranges are in nonce space, not W3. **Add:** `mine` aborts if a candidate fails `top32(hash) <= t7` on the reference re-hash (kernel bug); known answers for mainnet blocks 800000/850000/900000/910000 plus 3 recent blocks, all with rolled versions (the base version is offset so the answer sits at rolled index r=5); a thread-partition coverage test; a reference-vs-OpenSSL cross-check on random lengths; and a TSan build target |
| 4 | **[major]** T7's "scalar on idle ports p1/p6" is half wrong: scalar rotates/shifts use p06, and p0 is the vector bottleneck. Suggests a nonce-parallel zmm schedule table | **Partly disputed, to be settled by measurement.** The model assumed one scalar schedule per 16 hashes. The implementation shares each nonce's schedule across `VR_GROUPS` = 8 vectors (128 versions), which amortizes it to ~4 scalar instructions per hash. **Measure** `VR_GROUPS` = 1, 2, 4, 8 and GPR broadcasts in the VR loop. Build the vector-table variant only if the sweep shows the scalar schedule costing > 2% |
| 5 | **[major]** Cycles/hash at TSC frequency is wrong. The core runs at ~3.2 GHz scalar and ~2.7 GHz under zmm | **Accepted, confirmed independently** (2.7 GHz under AVX-512, 3.0–3.3 GHz scalar). Report **ns/hash** plus an estimated core-cycles/hash using a dependent-add-chain probe measured under the same ISA license (`fbm freq`) |
| 6 | **[major]** 5 repeats cannot resolve 1–4% effects on a noisy VM (one run dropped to 12.5 MH/s) | **Accepted.** Add `fbm bench --interleave`: ≥ 20 rounds, each round running every kernel once in rotating order, pinned, after a warm-up. Report median, IQR and min/max per kernel, plus a **bootstrap 95% CI** for each speedup ratio, paired by round. Ablations use deterministic op counts from the generator and objdump, not noisy timings |
| 7 | **[major]** Economics needs live numbers and a clear verdict | **Accepted.** `tools/economics.py` pulls live hashrate, difficulty, fees and price from mempool.space and prints revenue per day and per year, expected years per solo block, and J/TH against a current ASIC. The report states plainly that CPU mining is a pure loss |
| 8 | **[minor]** Rounds 57–60 need no `a` work; phatk compares against −(IV7+K60) | The rounds 57–60 point is handled: the generator's dead-code elimination removes all `a` work after round 56 (116 Σ0 = 60 + 56). The phatk constant trick saves one add per 16 hashes (0.04%). Noted, not worth the complexity |
| 9 | **[minor]** The two-vector interleave is counterproductive (register pressure) | **Dropped.** The measured 88–91% of ALU peak confirms that one stream is port-bound |
| 10 | **[minor]** Expected gains were overstated or used the wrong reference | Claims now come only from measurements against cpuminer-opt, with CIs |
| 11 | **[major]** Scope: cut the Python baseline and hand-written kernels; add a "milestone 0" harness | The generator already gives scalar/AVX2/AVX-512 from one source, so no kernel is hand-written. The Python baseline is dropped from the benchmark table |

## 2. Goal, restated honestly

Build the fastest SHA-256d search engine we can for this CPU, measure it
against the best existing CPU miners, and isolate exactly how much of the
speed is new (T7) and how much is prior art (T1–T5). Then show with live
numbers why this does not make CPU mining worthwhile, and say where real
efficiency comes from: ASICs at ~10–15 J/TH, cheap power, and version
rolling, which ASICs already use (14 of the last 15 mainnet blocks have
BIP 320 bits set).

## 3. Work items (in order)

1. **Baselines:** `bench/cpuminer-opt/` builds cpuminer-opt's sha256d
   16-way/8-way kernels (pinned commit, clang) into a harness with the same
   timing method as `fbm`. `bench/cpuminer/` does the same for pooler's AVX2
   assembly.
2. **Measurement:** `fbm freq` (core-clock probe for scalar/AVX2/zmm) and
   `fbm bench --interleave` (rounds, IQR, bootstrap CIs, ns/hash).
3. **Correctness:** the fatal candidate check in `mine`, more mainnet
   known answers, the partition test, the OpenSSL cross-check, and the TSan
   target.
4. **T7 validation:** the `VR_GROUPS` sweep, a GPR-broadcast audit of the VR
   loop, and the vector-table variant only if needed.
5. **Codegen guard:** `tools/asmcheck.py` wired into `make check`.
6. **Economics** from live data.
7. **Report:** `docs/RESULTS.md` and `README.md`, holding measured numbers
   only, each with a reproduction command.

## 4. Non-goals

* Stratum networking. It adds no efficiency, and version rolling in a pool
  would also need BIP 310 negotiation.
* SHA-NI kernels: this CPU has no SHA extensions, so the code could not be
  tested here. This goes under future work.
* GPUs, FPGAs, "quantum" speedups, and ML nonce prediction. The last one
  cannot work unless SHA-256 is broken.
