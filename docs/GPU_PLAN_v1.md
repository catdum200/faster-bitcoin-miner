# GPU plan v1: a SHA-256d kernel for the Radeon RX 9060 XT 16 GB

Status: first draft, written before any GPU code. An independent agent will
critique it (`GPU_CRITIQUE.md`), and the revised plan will be
`GPU_PLAN.md`. This repeats the CPU process (`PLAN_v1` → `CRITIQUE` →
`PLAN` → `RESULTS`).

## 0. Honest framing

- **The economics do not change.** The README's estimate for this card is
  ~3 GH/s, which is about $0.04 per year of expected revenue against
  $120-210 per year of electricity for the card alone. A solo block would
  come about once in 6 million years. No kernel can close a 2,000-4,000x
  energy gap to ASICs. This work is engineering, not income.
- **"Faster and more efficient" means one thing here:** fewer vector-ALU
  instructions per hash. The kernel has no memory traffic in its hot loop,
  so at a fixed clock and power limit, fewer instructions per hash means
  more hashes per second *and* more hashes per joule.
- **Constraint: no GPU is available here.** This VM has no AMD GPU, so
  nothing can be timed on the target. What we *can* do here:
  1. **Correctness.** Run the real OpenCL kernels and the real host code on a
     CPU OpenCL implementation (PoCL), against the reference SHA-256.
  2. **Exact static cost.** Compile the same kernels for the card's ISA
     (`gfx1200`) with clang/LLVM 20 and count the instructions in the hot
     loop.
  3. **A measurement harness.** Ship `fbm-gpu test` and `fbm-gpu bench`, so
     the owner of the card can verify and time everything with one command.

## 1. Target facts (RX 9060 XT = Navi 44 = gfx1200, RDNA4)

| fact | value | source / how checked |
|---|---|---|
| Compute | 32 CUs = 16 WGPs = 64 SIMD32 = 2048 lanes | AMD / Wikipedia specs |
| Clock | 2530 MHz game, 3130 MHz boost; TBP 160 W (16 GB) | AMD specs; sustained clock under integer load unknown |
| Integer VALU rate | 32 lanes/cycle/SIMD for 32-bit add/logic/alignbit/bfi/add3/xor3 (assumed full rate) | RDNA slides: "1 instruction per cycle for wave32"; **to verify** |
| Useful instructions | `v_alignbit_b32` (rotate), `v_xor3_b32`, `v_add3_u32`, `v_bfi_b32` (Ch), `v_xad_u32`, `v_perm_b32` (byte swap) | RDNA4 ISA guide opcode list; clang 20 emits them |
| **No 3-input logic op** | `v_bitop3_b32` exists only on gfx950+; llvm-mc rejects it for gfx1200 | tested |
| VOPD dual issue | pairs only FP ops, MOV/CNDMASK (X) with ADD/LSHL/AND (Y): two integer ops cannot pair | RDNA4 ISA §7.8 |
| Scalar ALU | one per SIMD, no rotate instruction; clang puts wave-uniform rotates on the **VALU** (`v_alignbit_b32 v, s, s, n`) | tested (clang 18, gfx1200) |
| Scalar loads | a uniform load from `const global uint *restrict` becomes `s_load_b128/b512` into SGPRs; VALU ops read SGPRs directly (≤ 2 per instruction) | tested (clang 20, gfx1200) |
| Issue | "each SIMD32 issues 1 instruction every cycle"; 5 cycles of exposed VALU latency, hidden by other waves | AMD RDNA architecture slides (GPUOpen) |
| Registers | 1536 VGPRs per SIMD (192 KB), max 16 waves/SIMD; SGPRs ≤ 106 per wave | RDNA4 notes (azhirnov/cpu-gpu-arch), ISA guide |
| Caches | I$ 32 KB, K$ (scalar) 16 KB, LDS 128 KB per WGP (64 KB per workgroup) | same |

Consequences for SHA-256 on this ISA, per operation:

| op | instructions |
|---|---|
| Σ0, Σ1 | 4: 3× `v_alignbit_b32` + `v_xor3_b32` |
| σ0, σ1 | 4: 2× `v_alignbit_b32` + `v_lshrrev_b32` + `v_xor3_b32` |
| Ch(e,f,g) | 1: `v_bfi_b32(e, f, g)` |
| Maj(a,b,c) | 2: `v_xor_b32(a, b)` + `v_bfi_b32(a^b, c, b)` |
| n-term sum | ⌈(n−1)/2⌉ with `v_add3_u32` |
| full round | ~15: Σ1 4 + Ch 1 + Σ0 4 + Maj 2 + adds 4 (h+K+W, T1, d+T1, T1+Σ0+Maj) |
| schedule word | ~10: σ0 4 + σ1 4 + 2 adds |

## 2. Cost model

Predicted hash rate = 2048 lanes × f / (VALU instructions per hash), with
f between 2.5 and 3.1 GHz (unknown sustained clock at the power limit).
Rough count for a nonce-per-lane kernel with every CPU trick (midstate,
folding, early exit):

- hash 1: ~61 rounds × 15 + ~46 schedule words × ~8 ≈ 1280;
- hash 2: ~61 rounds (57-60 without the a-path) + ~45 schedule words
  ≈ 1290;
- total ≈ **2550 VALU/hash → 2.0-2.5 GH/s**.

The generator will replace these estimates with exact counts.

## 3. Design

Everything comes from the existing partial evaluator (`gen/gen_kernels.py`),
with a new target `rdna`. Its output is OpenCL C: each work-item is one lane,
so vector ops become plain `uint` ops that the compiler maps to VALU
instructions. There is no hand-written SHA code.

| id | technique | status |
|---|---|---|
| G1 | Nonce-per-lane kernel with every prior-art trick: midstate; JOB values precomputed on the host and passed as kernel arguments (SGPRs); rounds 0-2 folded; early exit after round 60 of hash 2; `bswap(H7) <= t7` compare | prior art (poclbm/phatk/diakgcn, 2011-2013) |
| G2 | **Version rolling across lanes.** One work-item = one rolled version (BIP 320), and a workgroup shares one nonce at a time. The block-2 schedule of hash 1 depends only on the nonce, so it moves out of the per-lane code | new relative to GPU miners compared |
| G3 | **Where the shared schedule is computed.** A pre-pass kernel computes the generator's NONCE values (schedule + K) for a block of nonces into a table. The main kernel reads each nonce's row with scalar loads (3× `s_load_b512` per nonce per wave), and every VALU instruction takes the value as an SGPR operand | new |
| G4 | LANE values (midstate and rounds 0-2 per version) are computed on the host once per job and loaded into VGPRs once per work-item, then reused across its nonce loop | new |
| G5 | 3-input add trees in the generator (Huffman with arity 3), so sums map to `v_add3_u32` | codegen |
| G6 | Maj as `bfi(a^b, c, b)`, Ch as `bfi`, Σ via `rotate` + 3-way XOR | codegen |

Why G3 uses a table and scalar loads, not the scalar ALU: the SALU has no
rotate (3 ops each), clang moves uniform rotates to the VALU anyway, and the
slides suggest SALU instructions may compete with VALU for the SIMD's single
issue slot. A table costs 3 SMEM instructions per nonce per wave, against
~350-400 VALU instructions saved.

Why not LDS: it needs barriers and VGPRs for every loaded word, and scalar
loads need neither.

**Expected gain:** G2 removes the hash-1 schedule, ~15% of VALU work, for
~1.15-1.18x over G1. That is the same mechanism and size as on the CPU
(1.17x there).

Launch geometry:

- **G1:** one nonce per work-item. A launch covers ~2^28 nonces, ~100 ms.
- **G2:** a 2-D grid of (versions × nonce blocks). A work-group of 64-256
  consecutive versions loops over NPB nonces. NPB is chosen so that a launch
  is ~100 ms, well under the Windows 2 s TDR limit.

Hits go to a global atomic counter plus a buffer. Overflow is counted,
never dropped silently, as on the CPU.

## 4. Host program

`fbm-gpu`: C11 + OpenCL 1.2, no pthreads, no Linux-only calls. It builds on
Linux with `make gpu` and on Windows with MSYS2/MinGW. It reuses
`src/sha256.c` and `src/header.c`.

- `fbm-gpu list`: platforms and devices.
- `fbm-gpu test`: the correctness tests below, on the chosen device.
- `fbm-gpu bench`: interleaved rounds, median, IQR and bootstrap CIs, as
  `fbm bench`. It uses kernel time from OpenCL profiling events.
- `fbm-gpu mine --header HEX [--versions N]`: like `fbm mine`.

The kernel source is embedded in the binary.

## 5. Correctness

The same standard as the CPU kernels: the reference SHA-256 is the oracle.

- Exact candidate-set equality with the reference on random rectangles: loose
  and tight thresholds, odd sizes, the end of the nonce and version spaces,
  and t7 = 0.
- Rediscover the real (version, nonce) of the 9 mainnet blocks, with the base
  version offset so that the answer sits at r = 5.
- Candidate overflow counting.

These run here on PoCL, and on the real card with `fbm-gpu test`. `mine`
aborts on a candidate that fails its own filter.

## 6. Static performance audit

`tools/gpu_isacheck.py` (`make gpu-check`) compiles the kernels with
clang 20 for gfx1200. For the hot loop of each kernel it reports:

- VALU, SALU, SMEM, VMEM and `s_delay_alu` counts per hash;
- VGPR, SGPR and scratch (spill) usage, and hence occupancy;
- code size against the 32 KB I$.

It checks the VALU count against the generator's count, as `asmcheck.py`
does on the CPU.

## 7. Baselines

cgminer 3.7.2's OpenCL kernels (`poclbm`, `phatk`, `diakgcn`, `diablo`;
public domain / GPL-3), the last maintained open-source GPU sha256d miners.
They are fetched at a pinned tag, not vendored. They are compiled with the
same compiler for gfx1200, and their VALU/hash is counted by the same audit.
G1 should match the best of them; G2 is the claimed gain.

## 8. Economics

Update the README with the predicted rate range, using
`tools/economics.py --mhs <pred> --watts 160`.

## 9. Work items (in order)

1. Generator: add the `rdna` target, ternary add trees, OpenCL emission, and
   exact op counts.
2. OpenCL kernels G1 and G2 (plus the schedule pre-pass), from one template.
3. Host `fbm-gpu`, and the tests on PoCL.
4. ISA audit and cgminer baselines.
5. `docs/GPU_RESULTS.md`, README and CLAUDE.md.

## 10. Non-goals

- HIP/CUDA/Vulkan: OpenCL works with the stock Windows and Linux drivers.
- A Stratum client. Pool version rolling would also need BIP 310.
- Claiming any measured speed on the RX 9060 XT: there is none from this VM.
