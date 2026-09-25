# Critique of PLAN_v1

Evidence was gathered on this VM (scripts in `/tmp/fbm-critique/`). `build/harness.c` wraps cpuminer-opt's kernels, and `opmodel.py` counts uops with constant folding and dead-code elimination. `freq.c`, `contend.c` and `fe.c` are microbenchmarks. Port data is from uops.info (Cascade Lake).

**Verdict:** the SHA-256 algebra in T1–T4 is mostly right (`0xa41f32e7 = −IV7` is verified). But the plan re-derives 2011–2016 prior art, benchmarks against a strawman, measures cycles wrongly, and builds its one new idea (T7) on a false port assumption.

## Findings

**1. [critical] The baseline is a strawman, and T1–T5 are prior art.**
(a) T0 compares against naive, OpenSSL and Python hashing. Midstate, the 3-round prehash, W16–W31 pre-extension, zero-word folding and the early exit are all in pooler's cpuminer `sha256d_ms` (2011–13). cpuminer-opt ships all of them as a 16-way AVX-512 kernel.
(b) I built cpuminer-opt's kernels into a harness. It finds the genesis nonce. Single-thread results:

| kernel | MH/s |
|---|---|
| pooler scalar C | 2.3 |
| AVX2 8-way | 9.4 |
| AVX-512VL ymm 8-way | 18 |
| zmm 16-way, gcc | 23.1 |
| zmm 16-way, clang | **27.1** |

With 4 threads the gcc build reaches 90–96 MH/s. The uop model's floor for a T1–T4 kernel is 89 core cycles/hash. The clang build already runs at ≈99 cycles/hash (27.1 MH/s at the measured 2.7 GHz), about 90% of that floor.
(c)
- Headline baseline: clang-built cpuminer-opt `scanhash_sha256d_16way` on this VM.
- Keep naive/Python only as correctness oracles, and credit the prior art.
- Honest ceiling: about +10% from T1–T4 and at most +18% from T7.

**2. [critical] Compiler codegen matters as much as T7, and the plan ignores it.**
(a) "gcc 13" is not a neutral choice.
(b) On the same source, gcc emits 126 `mov $K,%eax; vpbroadcastd %eax,%zmm` pairs in the two hot functions. Each is an extra p5 uop (`VPBROADCASTD zmm,r32` = 1\*p5). clang uses `vpaddd K(%rip){1to16}` instead, which is one fused p05+p23 uop. Medians over 6 interleaved runs: 23.1 vs 27.1 MH/s, **+17%** for clang. For a value computed in scalar code, which is T7's design, **both** compilers emit `vpbroadcastd %edi,%zmm`. Neither produces a memory broadcast.
(c)
- Build with both compilers.
- Emit broadcasts through an inline-asm helper (`vpaddd %1%{1to16%},%0,%0`).
- Make CI fail if the hot loop contains a GPR `vpbroadcastd` or indexed addressing (which un-laminates).

**3. [critical] These gaps can drop solutions silently.**
(a, b)
- The test mask replaces the compare, so production's `e==0xa41f32e7` path is exercised only by 2 real blocks. Neither has rolled version bits.
- The early exit is *exact* (e60 = −IV7 ⇔ H7 = 0), so a candidate that fails the reference re-check is a **bug**. Under "fully verify candidates" it would be discarded as a false positive. That is how a byte-order bug loses blocks.
- The kernel iterates W3, the big-endian word, and the nonce is bswap32(W3): for genesis, W3 0x1dac2b7c → nonce 2083236893. A contiguous W3 range is not a contiguous nonce range, so `--start/--count` is ambiguous. No per-nonce bswap is needed in the hot loop.
- The H7==0 prefilter is invalid for targets ≥ 2^224 (regtest `0x207fffff`, or share difficulty < 1). For those it finds nothing.

(c)
- Use one compare in every mode: `vptestnmd(e60+IV7, mask)`, about 1 uop per 16 hashes.
- A kernel hit that fails the reference check aborts the run.
- Add version-rolled mainnet known answers. I verified these headers from mempool.space: block 800000 (version 0x341d6000, nonce 106861918), 850000 (0x2bbb8000) and 900000 (0x20aba000). The CLI must print exactly (version, nonce).
- Test the partial last vector, counts that are not multiples of 16/256, starts near 2^32−16, several hits in one vector, and thread-partition coverage. Run TSan on the result queue.
- Fall back to a full compare for easy targets, and cross-check the reference against OpenSSL on random lengths.

**4. [major] T7's "scalar on idle p1/p6" is half wrong, and a vector design is simpler.**
(a) Only scalar ALU ops can use p1/p6. `RORX/ROR/SHR r32` run on **p06**, and p0 is the vector bottleneck: `VPRORD/VPSRLD zmm` are p0-only, while `VPADDD/VPTERNLOGD zmm` use p05.
(b) The model puts the scalar schedule at 465 ops per 16 hashes, 228 of them rotates or shifts. The synthetic test used the SHA round mix (17 zmm uops, measured 8.4 cycles per round):
- adding 2 rorx + 2 ALU ops per round (about T7's ratio) cost ≤1%;
- adding 4 rorx cost −1% to +10% (noisy).

So the scalar design is feasible but fragile, and it depends on #2. The front end is not the limit (about 2.4 fused uops/cycle; a 3400-instruction unrolled body showed no decode penalty).
(c) Compute block-2 schedules for **16 nonces in one zmm pass**:
- Store W+K in a 64×16-dword table (4 KB).
- Run the 16 version lanes of each nonce with `vpaddd [p+64t]{1to16}`. Advance p by 4 per nonce; base+disp addressing keeps micro-fusion.
- Model: 75.6 cycles/hash, vs 74.7 with "free" scalar and 89.1 without T7. That makes 1.18× the **upper bound**.
- Keep ternlog constants in registers, because `VPTERNLOGD m32bcst` un-laminates into 2 slots.
- Pools need BIP 310 mask negotiation before version rolling is usable.

**5. [major] "Cycles/hash at TSC frequency" is wrong.**
(a) The TSC runs at 2.8 GHz; the core does not.
(b) `freq.c` timed a dependent-add chain: 3.16–3.28 GHz for scalar and ymm code, 2.68–2.78 GHz right after zmm bursts (AVX-512 license). TSC-based cycles are therefore off by −14% to +3%, and comparisons across ISAs by about 15–20%. zmm still wins on wall-clock (24 vs 18 MH/s for EVEX ymm).
(c) Report ns/hash, the core GHz measured under the same license, and core cycles/hash.

**6. [major] The statistics cannot resolve the effects being claimed.**
(a) The plan uses "5 repeats, median".
(b) Single-thread gcc runs ranged 21.8–24.6 MH/s, and one run fell to **12.5** (external load: the VM is shared). On a quiet machine the clang build still ranged 26.0–27.9. 4-thread runs ranged 90.1–95.6. T3, T4 and the micro-optimizations are each 1–4%.
(c)
- Do ≥20 interleaved A/B runs of 1–2 s after a ≥100 ms warm-up, with pinned threads.
- Report the median with an IQR or bootstrap CI, plus the minimum; flag outliers.
- Add deterministic ablations (uop counts from `objdump`).

**7. [major] The economics section needs numbers and a verdict.**
Live data from mempool.space today: 930 EH/s, difficulty 1.33e14, an average of 3.15 BTC per block, and a price of $83,839. At an optimistic 120 MH/s:
- 5.9e-11 BTC/day, about **$0.0018 per year**.
- Solo mining would take about **150 million years** per block. P(block within a year) ≈ 7e-9.
- Energy is about 2.5–3.3e5 J/TH (TDP estimate), against 13.5 J/TH for an S21 XP (spec): about 2×10^4 times worse.
- VM rent (about $0.1–0.2/h) exceeds revenue by roughly 5×10^5–10^6.

The report must say that CPU mining is a pure loss, that a software gain of at most 1.3× cannot change that, and that the project's value is educational. Cloud providers' terms often restrict mining too.

**8. [minor] T2/T3/T4 details lag behind kernels from 2011.**
- e60 depends on a53…a56 only, so the T2/a work in rounds **57–60** is dead, not just in round 60. cpuminer computes only the e-chain there.
- phatk.cl (2011) compares against −(IV7+K60) = **0x136032ED**, which saves the K60 add.
- Round 3 is 2 adds (one for a, one for e), not 1.
- Model gains: T2 5.8%, T3 2.9% (mostly from the zero schedule words; hash-2 round-0/1 folding is about 0.5%), T4 3.1% as planned vs 4.2% optimal.
- "The compiler can do it" only holds for fully unrolled code with visible constants, so check the asm.

**9. [minor] T8's two-vector interleave is counterproductive.**
Per round, the dependency chain is about 4 cycles (rot→ternlog→add→add) but port throughput needs about 8.5, so one stream is already port-bound (cpuminer-opt reaches about 90% of the floor). Two streams need about 2×(8 state + 16 W) > 32 zmm and would spill. Drop it, and keep the W window in memory (micro-fused loads on p23 are free).

**10. [minor] The expected gains are overstated or measured against the wrong reference.**
T5's "8–16× over scalar" measured 4.1× (AVX2) and 10.5–12× (zmm) against a scalar C kernel that already has T1–T4. T1's "1.5×" is relative to a baseline no miner uses. T6's "~4×" measured 3.8–4.0× here, which a VM cannot guarantee.

**11. [major] Scope.**
- **Cut:** the Python baseline, the optimized-scalar milestone, and the hand-written AVX2 kernel (template ymm/zmm from one source instead).
- **Add milestone 0:** the oracle and known-answer harness, the cpuminer-opt baseline, the frequency probe, and uop-count checks.
- **Then:** zmm parity with cpuminer-opt, then T7 (the only novel part: cpuminer-opt has no version rolling), then tuning.
- **Avoid:** hand-scheduling scalar T7, and a full ablation matrix across 4 ISAs.

## Top 5 changes

1. Use clang-built cpuminer-opt 16-way on this VM as the baseline. Credit the prior art, and restate the expected gain as ≤1.3×.
2. Force `{1to16}` memory broadcasts with inline asm. Build with gcc and clang, and fail CI on a GPR `vpbroadcastd` in the hot loop.
3. Implement T7 as a nonce-parallel zmm schedule table, not as scalar code.
4. Make production and tests use the same compare. Treat a reference mismatch as fatal. Add the rolled-version mainnet known answers (800000/850000/900000) and range/wrap tests.
5. Measure the core clock (2.7 vs 3.25 GHz). Use at least 20 interleaved runs with CIs. Write the economics from the live numbers above.
