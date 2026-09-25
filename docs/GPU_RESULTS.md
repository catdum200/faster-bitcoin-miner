# GPU results: SHA-256d on the Radeon RX 9060 XT (RDNA4, gfx1200)

**What this is, and what it is not.** No GPU was available: the work was done
on a 4-vCPU VM. So every number here is one of three kinds, and each is
labelled:

1. **Exact static counts.** The shipped OpenCL kernels, compiled for the
   card's ISA (gfx1200) by clang 20, with every instruction in the hot loops
   counted (`tools/gpu_isacheck.py`). The same audit runs on cgminer's kernels.
2. **Correctness runs.** The real kernels and the real host code, on a CPU
   OpenCL implementation (PoCL), and the Windows build under Wine.
3. **Predictions**, clearly marked. **No hash rate, efficiency or speedup has
   been measured on an RX 9060 XT.** `fbm-gpu report` measures all of them in
   one run on the card (see "What only the card can answer").

The raw outputs are in [`bench/results/`](../bench/results/) (`gpu-*.txt`).
The process was the CPU one: [`GPU_PLAN_v1.md`](GPU_PLAN_v1.md) →
[`GPU_CRITIQUE.md`](GPU_CRITIQUE.md) (an independent agent, with its own
prototypes) → [`GPU_PLAN.md`](GPU_PLAN.md) → this file.

## Headline (static, exact)

Per hash = per hot-loop iteration of one lane (work-item). "All instr." counts
every instruction as if it took the SIMD's single issue slot: scalar ALU,
scalar loads and the `s_delay_alu` scheduling hints. That is the pessimistic
case. Rates assume 2048 lanes at 2.53-3.13 GHz (the card's game and boost
clocks), and a full-rate VALU.

| kernel | layout | VALU/hash | all instr./hash | VGPRs → waves/SIMD | loop bytes | predicted GH/s |
|---|---|---:|---:|---:|---:|---:|
| cgminer `poclbm` (2013) | 1 nonce per work-item | 2578 | 2907 | 40 → 16 | 20480 | 1.8-2.5 |
| cgminer `phatk` | 〃 | 2575 | 2899 | 40 → 16 | 20480 | 1.8-2.5 |
| cgminer `diakgcn` | 〃 | **2572** | 2851 | 40 → 16 | 20224 | 1.8-2.5 |
| cgminer `diablo` | 〃 | 2574 | **2812** | 41 → 16 | 20096 | 1.8-2.5 |
| `fbm_nonce` (this repo) | nonces in lanes | 2580 | 2688 | 43 → 16 | 19720 | 1.9-2.5 |
| **`fbm_vr` (this repo)** | **rolled versions in lanes, shared schedule** | **2196** | **2445** | 67 → 16 | 17332 | **2.1-2.9** |

- **`fbm_vr` needs 1.171x fewer vector instructions per hash than the best
  prior-art kernel** (diakgcn). If every scalar and delay instruction also
  costs an issue slot, the figure is **1.150x** (against diablo, the best on
  that metric).
- The expected speedup on the card therefore lies between those two figures,
  **~1.15-1.17x**, if the VALU runs at full rate. It is unmeasured.
- Against our own nonce kernel the range is wider, 1.10-1.17x, because
  `fbm_nonce` has unusually few scalar instructions.

What this says:

- **Version rolling is the only real gain.** Midstate, precomputation and the
  early exit after round 60 are 2011-2013 prior art. Modern LLVM compiles
  cgminer's hand-tuned kernels and our generated nonce kernel to the same
  count, 2572-2580 VALU per hash, within 0.3%. The prior-art tricks are now
  simply what the compiler produces.
- **The shared schedule really is shared.** In the `fbm_vr` loop the whole
  block-2 message schedule of the first hash (47 words, with the round
  constants folded in) arrives through **5 scalar loads per nonce per wave**.
  There are no vector memory instructions, and every VALU instruction reads
  it as a free SGPR operand. It is computed once per nonce by a pre-pass
  kernel (463 VALU) and shared by every rolled version: ~0 per hash.
- **Nothing else limits it statically.**
  - No spills; the maximum 16 waves per SIMD.
  - A 17 KB loop, inside the 32 KB instruction cache.
  - The generator's instruction count predicts the compiled loop within
    +1.0%.
  - RDNA4 has no 3-input logic op (`v_bitop3` is gfx950-only) and no integer
    dual issue (VOPD), so SHA-256 costs what the table shows. Ch is one
    `v_bfi_b32`, Maj is two instructions, each Σ is four.

## Why this design (and not three others)

The mechanism is the CPU's (overt AsicBoost, BIP 320): block 2 of the first
hash depends on the nonce but not the version. So lanes that hold different
rolled versions of the same nonce share its schedule. On a GPU the question
is where that shared work lives. The first two rows are the critic's
prototypes, compared with its own nonce-layout build (2585 VALU); the last
row is the shipped kernel, compared with `fbm_nonce` (2580):

| where the shared schedule is computed | VALU/hash | vs nonce layout |
|---|---:|---:|
| inline, wave-uniform: clang puts uniform rotates on the VALU (the SALU has no rotate) | 2693 | 0.96x (a loss) |
| per wave, 32 nonces in 32 lanes, then 47 `v_readlane` per nonce | ~2268 | 1.14x, occupancy 12 |
| **pre-pass table, read with scalar loads (shipped)** | **2196** | **1.175x** |

Other choices:

- **LDS** would need ≥ 12 `ds_load_b128` into VGPRs per nonce and saves
  nothing over scalar loads.
- **LANE values:** each version's midstate, and the parts of rounds 0-2 that
  follow from it, take 19 VGPRs. They are loaded once per work-item and
  reused for its whole nonce loop.
- **Midstates:** the host computes the 65,536 midstates once per job, in
  23 ms here.
- **Few versions.** The pre-pass costs 463 VALU per row, spread over the
  versions that share the row, so 463 / versions per hash. That is 0.7% even
  when only one wave (32 versions) shares it; the CPU needs ≥ 64 versions.
  (The critique's "~0.5 VALU per hash" divided by 32 once too often.) With
  the default 64-wide work-groups, rolling a multiple of 64 versions leaves no
  lanes idle; `mine` switches to `fbm_vr` from 64 versions.

## What each technique is worth (exact counts, gfx1200)

`python3 gen/gen_kernels.py --ablation`: vector instructions per hash, per
work-item, from the generator's cost model. The compiler adds +1.0%.

| step | per hash | vs previous | total reduction |
|---|---:|---:|---:|
| naive: 3 compressions per nonce, nothing precomputed | 4250 | - | - |
| + midstate: 2 compressions per nonce | 2815 | -33.8% | 1.51x |
| + constant folding / precompute | 2665 | -5.3% | 1.59x |
| + early exit after round 60 of hash 2 | 2555 | -4.1% | 1.66x |
| + version rolling (schedule shared by all rolled versions) | 2175 | -14.9% | 1.95x |

## How robust the static numbers are

- **Compiler.** clang 18 and clang 20 give identical VALU counts: 2580 and
  2196.
- **Architecture.** RDNA2 (gfx1030), RDNA3 (gfx1100) and both RDNA4 chips
  (gfx1200, gfx1201) give the same VALU counts. RDNA2 has no `s_delay_alu`,
  so on it even the pessimistic ratio against the best prior art is 1.181x
  ([`gpu-isa-other-archs.txt`](../bench/results/gpu-isa-other-archs.txt)).
- **The real driver is a different compiler.** On Windows or ROCm, the
  driver's own LLVM fork compiles the source. `fbm-gpu dump` saves what it
  produced, and `tools/gpu_isacheck.py --binary` audits it with the same
  checks. `make gpu-codeobj` plus `fbm-gpu --program` can instead run the
  exact audited binary, where the runtime accepts it (ROCm).

## Correctness

| what | where | result |
|---|---|---|
| The generated GPU code, run one lane at a time on the CPU (`rdna-emu`, `rdna-emu-vr`): exact candidate sets vs the reference on random rectangles; the real (version, nonce) of 9 mainnet blocks; thread partitioning | `fbm test`, also under ASan+UBSan and TSan | pass |
| The OpenCL kernels and host, 7 differential cases × 2 layouts × 2 launch configurations (default, and 1000-hash launches with 3/7 nonces per work-item) | `fbm-gpu test` on PoCL | pass |
| 9 mainnet blocks, answer at rolled index 5 | 〃 | pass |
| **Planted answers at production geometry**: a real block with its answer at the highest version index (65535) and the last nonce of a multi-launch scan | 〃 | pass |
| Candidate overflow is counted, never dropped | 〃 | pass |
| The mining **canary** catches a faulty device (fault injected into the job buffer) | 〃 | pass |
| cgminer's poclbm kernel with our clean-room precompute finds all 9 blocks | 〃 | pass |
| The Windows build (`dist/fbm-gpu.exe`), whole suite | Wine 9 → PoCL | pass |
| The host code under ASan+UBSan | PoCL | pass |

The one thing these runs cannot catch: a miscompile by AMD's driver, or an
unstable card (heat, overclock, undervolt). So:

- `fbm-gpu test` and `fbm-gpu report` run the same suite on the card;
- `fbm-gpu mine` checks a known candidate set at start-up and every minute,
  and stops with a hardware-error message if it ever comes back wrong. The
  host re-hash of candidates catches false positives, but only these checks
  catch missed blocks.

## What only the card can answer

One command, `fbm-gpu report` (on Windows, `dist\fbm-gpu.exe report`; see
[`dist/README.md`](../dist/README.md)), writes one file:

| open question | how the report answers it |
|---|---|
| Are `v_alignbit`, `v_xor3`, `v_add3`, `v_bfi`, `v_xad` full rate on RDNA4? The ISA guide gives no rates | **issue-rate probe**: 8 independent chains of exactly one instruction each (inline asm), rate relative to `v_add_nc_u32` |
| Do scalar-ALU and `s_delay_alu` instructions take the vector issue slot? This decides whether `fbm_vr` gains ~1.17x or ~1.10-1.15x | probe mixes: 64 `v_add` with 64 `s_xor_b32`, or with 64 `s_delay_alu` |
| The shader clock under this load | the `v_add` probe's rate / 2048 lanes |
| Real hash rates, and the speedup over prior art | interleaved benchmark (10 rounds, bootstrap 95% CIs): cgminer's poclbm vs `fbm_nonce` vs `fbm_vr`, at 65,536 and at 64 versions per nonce |
| Sustained rate, board power, J/TH | 5-minute runs per layout. Board power and clock are logged on Linux (hwmon); on Windows, read them from Adrenalin's overlay |
| What the driver compiled | the dumped binary, for `gpu_isacheck.py --binary` |

Launches default to 8 ms, so a desktop on the same card stays responsive.
`--dedicated` uses 100 ms launches for a card without a display.

## Economics: still a bad idea

`tools/economics.py --mhs 2300 --watts 160 --usd-per-kwh 0.15` with live
data (block 968579, 924 EH/s, $84,007/BTC):

| quantity | value |
|---|---|
| predicted rate (middle of the `fbm_vr` range) | 2.3 GH/s |
| expected revenue | **$0.034 per year** |
| electricity at 160 W, $0.15/kWh | **$210 per year** |
| solo: expected time per block | **7.9 million years** |
| energy per hash vs an Antminer S21 XP | **~5,200x worse** (2,000-6,000x across 2-3 GH/s and 90-160 W) |

**Verdict: mining bitcoin on this card loses money at any kernel speed.** A
1.17x kernel moves revenue from about three cents a year to about four. On
this card the power limit is a bigger efficiency lever than the kernel: an RX
9070 kept 93% of its performance at 70% power, +32% performance per watt in
a gaming test. The report explains how to measure that for hashing, but it
does not change the verdict either. The kernel is worth it as engineering,
not as income.

## Where the critique was right

| critique finding | outcome |
|---|---|
| #1 no way to turn counts into GH/s or J/TH | **Right.** Built `fbm-gpu report`: tests, issue-rate probe, interleaved A/B with a real prior-art kernel, sustained runs with power logging, and the driver binary. Nothing here claims a measured rate |
| #2 the gain is 1.10-1.17x, not 1.15-1.18x | **Right against our own nonce kernel** (1.099x-1.175x). Against the best prior art, whose kernels carry 108-123 scalar instructions per hash, it is 1.150x-1.171x. The probe mixes decide which end holds |
| #3 energy efficiency is never measured; the power limit matters more | **Right.** No J/TH claim without logged power; the power-limit sweep is documented |
| #4 the audit checks clang's output, not the driver's | **Right.** Added `dump` + `--binary`, a ±2% tolerance, and the optional `--program` |
| #5 100 ms launches freeze a desktop | **Right.** Adaptive 8 ms launches, a 64-nonce loop cap, `--dedicated`, `mine --seconds`, Ctrl-C |
| #6 the README's ~3 GH/s is above the model | **Right.** The README now says ~2-3 GH/s (compile-based) and 2,000-6,000x |
| #7 G1 duplicates cgminer; porting all four costs too much | **Right.** Four static counts; one port (poclbm) with a clean-room precompute, verified on 9 blocks |
| #8 3-input add trees add nothing | **Right about speed:** binary trees compile to 2 fewer VALU (0.08%). The trees are kept as the cost model that lets the audit predict the compiled count within 1% |
| #9 G3's rationale was partly wrong | **Right.** The shipped table beats inline (0.96x) and `v_readlane` (1.14x). One correction to the critique: the pre-pass costs 463 / versions VALU per hash (0.7% at 32 versions), not ~0.5 |
| #10 PoCL tests neither the card's compiler nor its hardware | **Right.** Added planted production-geometry tests, a canary during mining, and a fault-injection test of the canary |
| #11 Windows is untested | **Right.** Run-time OpenCL loading, `make gpu-win`, and a prebuilt `dist/fbm-gpu.exe` that passes the suite under Wine |
| #12 fact details | All adopted: 96-bit instruction counts, the I$ check, wave size from the binary |

## Reproduce

```sh
make gpu && ./fbm-gpu test                 # tests on any OpenCL device (PoCL works)
bench/gpu-baselines/fetch.sh               # cgminer 3.7.2 kernels (not vendored)
python3 tools/gpu_isacheck.py              # gfx1200 audit: needs clang >= 18 with AMDGPU
python3 tools/gpu_isacheck.py --mcpu gfx1100
python3 gen/gen_kernels.py --ablation
make gpu-win                               # dist-style Windows build (mingw-w64)
./fbm-gpu report                           # on the real card: everything else
```
