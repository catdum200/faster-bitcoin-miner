# Critique of GPU_PLAN_v1

This VM has no GPU, so all evidence is static or comes from a CPU OpenCL run. Scripts and outputs are in `/tmp/claude-0/gpu-critique/`:

- `emit_cl.py` turns the repo's own generator (`gen/gen_kernels.py`, unchanged) into OpenCL kernels:
  - G1: one nonce per lane;
  - G2: one version per lane, with the schedule table read by uniform loads;
  - two other G2 splits: the schedule computed inline, and (`emit_rl.py`) a per-wave schedule broadcast by `v_readlane`.
- Compiler: `clang-20 -x cl -cl-std=CL1.2 -target amdgcn-amd-amdhsa -mcpu=gfx1200 -O3 -nogpulib`.
  - `isacount.py` counts each kernel and its hot loop from `llvm-objdump`.
  - Occupancy comes from the `; Occupancy:` lines of `-S`.
- `host.c` runs G1 and G2 on PoCL and checks each candidate set against `src/sha256.c`.
- `cgm_pre.h` compiles cgminer 3.7.2's four kernels with plain rotate/bitselect (no BITALIGN/BFI_INT patching).
- `rdna4.txt:N` means line N of the RDNA4 ISA guide text.

**Verdict:** the plan's arithmetic is right, and G2 is the right idea. Compiled for gfx1200:

- G1 needs **2585** VALU instructions per hash (the plan estimated ~2550).
- G2 needs **2206**, 1.17x fewer. That matches the claimed 1.15-1.18x.
- The table-plus-scalar-load split beats the three other splits I built.
- G1 is a pure reproduction: all four cgminer kernels compile to 2579-2582.

The problems are in what the count is turned into. "GH/s" and "more energy-efficient" depend on things nobody has measured: full-rate issue of integer VOP3 ops, the sustained clock, and power. The only channel for measuring them is one run by the card's owner, and the plan has not designed for that. Given the uncertainties below, G2's real gain is **1.10-1.17x**, and it depends more on occupancy than G1 does. The plan never measures power. It also ignores the largest efficiency lever on this card, the power limit and voltage, which is outside the kernel. 100 ms launches will make the desktop stutter. The README's ~3 GH/s is above the plan's own model.

## Measured: compiled hot loops (gfx1200, clang 20, per hash)

| kernel | VALU | other instr. (of which `s_delay_alu`) | loop bytes | VGPRs → waves/SIMD | vs G1, VALU only | vs G1, every instr. = 1 issue slot |
|---|---:|---:|---:|---:|---:|---:|
| cgminer poclbm / phatk / diakgcn / diablo (whole kernel) | 2582 / 2580 / 2579 / 2579 | | 19.7-20.0 KB | 40-41 → 16 | ~1.00 | |
| G1 nonce per lane | **2585** | 119 (97) | 19.6 KB | 43 → 16 | 1 | 1 |
| G2 table, uniform `s_load` | **2206** | 261 (217) | 17.2 KB | 65 → 16 | **1.172** | **1.096** |
| G2 table without `restrict` (becomes `global_load`) | 2205 | 282 (229) | 17.4 KB | 64 → 16 | 1.172 | 1.087 |
| G2 schedule computed inline, wave-uniform | 2693 | 101 (62) | 19.5 KB | 62 → 16 | 0.960 | 0.968 |
| G2 schedule for 32 nonces in 32 lanes, then 47 `v_readlane` per nonce | 2253 + ~15 | 206 (175) | 17.4 KB | 113 → 12 | 1.14 | ~1.09 |

The working tree's uncommitted `rdna` generator output compiles to the same numbers: 2596 (`fbm_nonce`) and 2212 (`fbm_vr`) VALU per hash. Its pre-pass costs 468 VALU per 32 table rows.

## Findings

**1. [critical] The instruction count is solid, but the plan has no way to turn it into GH/s or J/TH.**
(a) §2: rate = 2048 lanes × f / (VALU per hash), with f "between 2.5 and 3.1 GHz". §1: integer VALU at full rate is "assumed, to verify". §0: fewer instructions means more hashes per joule.
(b)
- **The per-op costs hold.** `v_alignbit_b32` appears exactly 1020 times per hash: 3 per Σ (236 of them) plus 2 per σ (156). `v_lshrrev` appears 156 times, and Maj is `v_xor` + `v_bfi`. clang also folds 94 final Σ XORs into adds (`v_xad_u32`), so "15 per round" is, if anything, pessimistic.
- **Nothing documents the issue rate.** The ISA guide has no throughput data: a search for "quarter rate", "half rate", "full rate" and "throughput" finds nothing about VALU. The only source for "1 instruction per cycle" is the RDNA1 slides (`rdna_arch.txt:122`). Full rate for `v_alignbit`, `v_add3`, `v_xor3`, `v_bfi` and `v_xad` on RDNA4 is an assumption.
- **f is unknown.** The README's own data point is hashcat at ~89 W on the GPU sensor, well under the 160 W board power. So an integer-only load may run near the 3.13 GHz boost clock, or it may not. Partner cards also differ (for example, a Sapphire Nitro+ is rated at 170 W).
- **The planned harness measures none of this.** `fbm-gpu test` and `bench` measure no clock, no power and no per-opcode rate, and they do not return the ISA the driver actually produced. The author cannot iterate on the card, so one run by the owner has to answer every open question.

(c) Add a milestone 0: **`fbm-gpu report`**, one command that writes one file for the owner to send back. It should contain:
- device, driver and OpenCL versions;
- the full test suite;
- a probe: long independent chains of each opcode the kernels use, reported as instructions per clock per SIMD, the GPU analogue of `fbm freq`;
- the interleaved bench;
- a sustained run of ≥ 5 minutes per kernel, logging clock and power;
- the driver's binaries (`CL_PROGRAM_BINARIES`) for audit here (finding 4).

Also have the owner run `hashcat -b -m 1400` once, as an independent calibration of the VALU rate.

**2. [major] G2's gain is 1.10-1.17x, not 1.15-1.18x, and it is more fragile than G1.**
(a) §3: "G2 removes ~15% of VALU work, ~1.15-1.18x", "the same mechanism and size as on the CPU".
(b)
- **Non-VALU instructions.** The VALU cut is real: 2585 → 2206, 1.172x. But G2 carries 261 non-VALU instructions per hash against G1's 119, mostly `s_delay_alu` (217 vs 97). The ISA says `S_DELAY_ALU` "may be executed in zero cycles" (`rdna4.txt:3462`), and the SQ snapshot register reports the issued instruction *types* as a bitmask (`rdna4.txt:1833-1841`), which suggests SALU and VALU from different waves co-issue. Neither is measured. If every instruction took an issue slot, the gain would be **1.096x**.
- **Occupancy.** G2 removes exactly the independent schedule work that filled G1's latency gaps. The compiler marks 115 dependencies at distance 2 in G2 (first `instid0` of each `s_delay_alu`), against 16 in G1. Taking those marks and the 5-cycle VALU latency (RDNA1 slides; Chips and Cheese measured ~5 cycles on RDNA4), a *single* wave per SIMD would run G2 at **0.94x** of G1. clang reports 16 waves/SIMD for both kernels, which hides this, but that holds only if the driver's compiler agrees.
- **On the CPU**, out-of-order execution hid this. Here only occupancy does.

(c)
- Quote the gain as 1.10-1.17x until it is measured on the card.
- Make the audit fail if the *driver's* binary has fewer than ~4 waves/SIMD or any scratch usage.
- Bench G2 against G1 as an interleaved A/B with CIs. Report the non-VALU counts next to the VALU counts.

**3. [major] "More energy-efficient" is claimed but never measured, and the biggest lever is left out.**
(a) §0: fewer instructions means more hashes per joule "at a fixed clock and power limit". §4: bench uses kernel time from profiling events.
(b)
- **No power data.** Nothing in the plan reads power. On Linux, amdgpu exposes `power1_average`, `power1_input`, `power1_cap` and `freq1_input` in hwmon (`drivers/gpu/drm/amd/pm/amdgpu_pm.c`, lines 3591-3626). On Windows this needs ADLX or a tool such as HWiNFO.
- **Short runs.** GPUs boost first and settle as they heat, so 1 s interleaved rounds give valid *ratios* but not absolute GH/s or J/TH.
- **Wall clock.** Profiling-event time leaves out launch gaps, the schedule pre-pass, and the host-side midstate work that G2 adds.
- **The power limit dominates.** Club386 measured an RX 9070 at a −30% power limit and −100 mV: 93% of its performance at 171 W instead of 244 W, **+32% performance per watt**. That is about twice G2's gain, and it costs no code. It was a gaming load, so the owner has to measure it for hashing.

(c)
- Report J/TH only from logged board power over a sustained run, and report hashes/s from wall-clock time over several launches.
- Add `--power-cap` sweep instructions: Linux `power1_cap`, or the Adrenalin power-limit slider.
- State plainly that power tuning, not the kernel, is the main efficiency knob on this card, and that neither changes the economics.

**4. [major] The audit checks a different program, from a different compiler, than the card runs.**
(a) §6: `gpu_isacheck.py` compiles with clang 20, and "checks the VALU count against the generator's count".
(b)
- **Different compiler, different source.** The driver (Adrenalin or ROCm) compiles the embedded source with its own LLVM fork. The audit build needs `-nogpulib` shims for `get_group_id`, `atomic_inc` and friends, so it is not even the same source.
- **The ISA is not guaranteed.** The plan asserts that `s_load` and SGPR operands will appear; only the driver's binary can show it.
- **What does not matter.** Without `restrict`, clang turns the table loads into 13 `global_load`s per nonce. The VALU count is unchanged (2205) and so is occupancy (16). So `restrict`/noclobber is not a performance cliff, and the "3× `s_load_b512`" expectation is wrong anyway: clang emits b512, b512, b256, b128 and b96 with 4 `s_wait_kmcnt` spread through the body.
- **What does matter**: scratch spills, low occupancy, a different wave size, and a missing `v_bfi`/`v_alignbit` pattern match. None of these can be seen from here.

(c)
- `fbm-gpu report` dumps `CL_PROGRAM_BINARIES`, and `gpu_isacheck.py` disassembles that file here with `llvm-objdump --mcpu=gfx1200`. If the Windows binary is not a plain AMDGPU ELF, say so.
- Check the generator's count with a tolerance (±2%), not for equality.
- Optionally try `clCreateProgramWithBinary` with the clang-built code object. If the runtime accepts it and the tests pass, the audited ISA *is* the executed ISA.

**5. [major] 100 ms launches will freeze a desktop, and G2 makes waves live as long as the launch.**
(a) §3: launches of "~100 ms, well under the Windows 2 s TDR limit".
(b)
- **TDR is the wrong bar.** A home RX 9060 XT usually also drives the display. cgminer's default intensity "d" exists "to maintain desktop interactivity", and `--gpu-dyninterval` targets **7 ms** per launch (cgminer 3.7.2 README and GPU-README).
- **Wave lifetime.** In G1 a wave lives for one hash, about 15 µs at 16 waves/SIMD. In G2 each work-item loops over NPB nonces, so every wave lives for the whole launch. Where the OS can only preempt between waves, a 100 ms G2 launch is a 100 ms display stall.

(c)
- Make the launch size adaptive: ~5-10 ms by default, larger only with `--dedicated`. Launch overhead at 10 ms is well under 1%.
- Cap G2's per-wave nonce loop (a few hundred nonces) and add nonce blocks in grid dimension 1 instead.
- Default `mine` to a time limit, and handle Ctrl-C.

**6. [major] The plan's predicted rate and the README's disagree.**
(a) §0 repeats the README's "~3 GH/s". §2 predicts 2.0-2.5 GH/s for G1.
(b)
- From the compiled counts, G1 = 2048 × (2.53-3.13 GHz) / 2585 = **2.0-2.5 GH/s**.
- G2 = **2.3-2.9 GH/s** if only VALU counts, or 2.1-2.6 if every instruction takes a slot.
- The README's ~3 GH/s is at or above the top of the best case. Its "1.5-1.8 compressions per attempt" divisor is optimistic: G1 is two nearly full compressions of ~1290 VALU each.
- `tools/economics.py --mhs 2000 --watts 160 --usd-per-kwh 0.15 --rent-usd-per-hour 0` (live data: height 968576, 926 EH/s) gives $0.029/year, 9.0 million years per block and 8.0e4 J/TH, which is **5.9e3x** an S21 XP. At 3000 MH/s: $0.044, 6.0 million years, 4.0e3x.

(c) In both the README and the plan, write "~2-3 GH/s (compile-based estimate, unmeasured)" and "2,000-6,000x worse than an ASIC". The verdict does not change.

**7. [major] G1 and the cgminer baselines duplicate each other, and the baselines cost more to port than the plan says.**
(a) §3 G1 "prior art"; §7: "fetch all four cgminer kernels, count them with the same audit, G1 should match the best".
(b)
- **Identical counts.** Modern LLVM erases the 2011-13 hand tuning: poclbm 2582, phatk 2580, diakgcn 2579 and diablo 2579 VALU per hash, against G1's 2585. G1 cannot beat them by more than ~0.3% (the phatk constant compare, `0x136032ED`).
- **The host precompute is GPL-3.** Running the baselines on the card needs cgminer's `precalc_hash()` (`findnonce.c`, GPL-3, ~80 lines) plus four different argument layouts from `driver-opencl.c`.
- **Their compare cannot be tested the same way.** The kernels compare only H7 == 0 (`== 0x136032edU`), so the plan's loose-threshold rectangle tests cannot run on them unmodified. They also write hits non-atomically (`output[output[FOUND]++]`).
- **Newer prior art exists.** ccminer's CUDA `sha256d` (tpruvot, 2017) is newer than cgminer. It uses a midstate only, and it is NVIDIA-only.
- **G2's novelty holds as far as I could check.** A GitHub code search (`asicboost extension:cl`) found one kernel, `chehw/spv-node` `hash256-asicboost.cl` (2021). Despite its name, it rolls the timestamp per work-item and hashes both blocks in full.

(c)
- Keep G1: it is free from the generator and is the same-harness nonce-lane reference.
- Port **one** baseline (diakgcn or poclbm, both public domain) with a clean-room precompute, and test it on the 9 known blocks with their true versions.
- Keep the other three as static counts only.

**8. [minor] G5 (arity-3 add trees in the generator) adds nothing.**
(a) §3 G5: "3-input add trees... so sums map to `v_add3_u32`".
(b) From plain binary Huffman trees, clang already forms 408 `v_add3_u32` and 94 `v_xad_u32` per hash (G1: 2585, G2: 2206). The working tree's explicit `add3` trees compile to 2596 and 2212, marginally *worse*.
(c) Drop G5. Keep the generator target-neutral, and let the audit report the fusion.

**9. [minor] G3's reasoning is partly wrong, but its choice is right, and it is better than the plan says.**
(a) §3: SALU instructions "may compete with VALU for the SIMD's single issue slot"; LDS rejected; "3 SMEM per nonce per wave".
(b)
- The SQ snapshot's per-type issue bitmask (`rdna4.txt:1833-1841`) weakens the "single slot" argument. The real reason is cost: SALU has no rotate. The ISA does have `S_LSHR_B64`, so each rotate is one 64-bit shift of a duplicated register pair, and portable OpenCL puts uniform rotates on the VALU anyway. The inline-uniform variant's loop has only 14 SALU instructions. Its uniform rotates are `v_alignbit_b32` with SGPR inputs, and after the first one the whole schedule runs on the VALU, at 0.960x of G1.
- `v_readlane` costs +2.8% VALU and drops occupancy to 12. LDS needs ≥ 12 `ds_load_b128` into VGPRs per nonce per wave and saves no VALU over `s_load`.
- A benefit the plan misses: the pre-pass costs 468 VALU per 32 rows, ~0.5 VALU per hash even when only 32 versions (one wave) share a row. So unlike the CPU (which needs ≥ 64 versions), **G2 pays off from 32 versions**, which is friendlier to pools with narrow BIP 310 masks.

(c) Keep the table. Fix the rationale. In the audit, check "no VMEM in the loop except the hit path" and "SMEM ≤ 8", not a specific load shape.

**10. [major] Testing: PoCL works, but it tests neither the card's compiler nor its hardware.**
(a) §5: the tests "run here on PoCL, and on the real card with `fbm-gpu test`".
(b)
- **What works here.** PoCL 5.0 (LLVM 16, 4 vCPUs) runs both prototypes. With t7 = 0x00ffffff, G1 reported 979 of 979 candidates in 262,144 hashes, and G2 reported 240 of 240 in 512 versions × 128 nonces, both exact. It hashes at 5-6.5 MH/s after a ~5 s cold compile, so test rectangles of ~2^24 are affordable.
- **What PoCL cannot catch:**
  - AMDGPU codegen or driver miscompiles;
  - wave-level behaviour (uniform loads, EXEC-masked hit branches, wave size);
  - hit-counter contention with ~32k live work-items;
  - watchdog and preemption behaviour;
  - wrong results from an unstable card (heat, or undervolting per finding 3).
- `mine`'s host re-check catches false positives. Only tests catch false *negatives*.

(c)
- On the card, also run production-geometry launches with a known answer planted in the last work-group, the last nonce block and the highest version index.
- During `mine`, run a periodic **canary**: a small launch with a loose t7 whose full candidate set is compared with the host reference.
- When the canary or a re-check fails, report a hardware error ("if you changed clocks or voltage, this is the likely cause") and stop.

**11. [minor] "Builds on Windows with MSYS2/MinGW" is untested, and it asks a lot of a home user.**
(a) §4: C11 + OpenCL 1.2, no Linux-only calls, MSYS2/MinGW.
(b) This VM has no MinGW toolchain, but Ubuntu's `mingw-w64` (11.0.1) and `gcc-mingw-w64-x86-64-posix` (13.2) are one `apt-get` away. The reused `sha256.c` and `header.c` include only `<string.h>`/`<stdint.h>`. Known MinGW pitfalls are `%llu` without `__USE_MINGW_ANSI_STDIO`, and timing (use `timespec_get` or `QueryPerformanceCounter`).
(c)
- Ask the owner for their OS and driver.
- Cross-compile `fbm-gpu.exe` here in `make gpu-win` and ship it, so the owner needs only the Adrenalin driver.
- Consider loading `OpenCL.dll` at run time, so no SDK or import library is needed.

**12. [minor] Details of the facts table.**
- **Confirmed:**
  - VOPD pairs no two integer ops: the X opcodes are FP ops plus MOV and CNDMASK (`rdna4.txt:4960-4968`); integer ADD/LSHL/AND exist only as Y opcodes (`:4971-4982`), and VOPD is "legal only for wave32" (`:4890`).
  - At most two scalar values (SGPRs or literals) per VALU instruction (`:4219`).
  - The SALU has no rotate.
  - `v_bitop3` is absent.
  - LLVM models gfx1200 with a 1536-VGPR file: 113 VGPRs give 12 waves, as on gfx1100, against 8 on gfx1102.
- **Add:**
  - "VOP3 plus a literal makes a 96-bit instruction and excessive use ... may reduce performance" (`:4014-4015`). G1 has 125 such instructions per hash (4.8%); the audit should report the count.
  - The I$ is not a limit: 17-20 KB loops fit in 32 KB. At ~7.6 B per VALU, fetch is well within the RDNA whitepaper's "32B ... every clock to each of the SIMDs".
- **Wave64 is not a throughput risk** for these kernels. It executes each VALU "twice" (`:837`), and no integer VOPD pair exists. Record `.wavefront_size` from the dumped binary anyway.
- "The kernel has no memory traffic in its hot loop" is not true of G2, which reads the table (192 B per nonce per wave). The traffic is harmless, but say so.

## Top 5 changes

1. **Design for one round trip to the card.** Ship a cross-compiled binary with `fbm-gpu report`: tests, a per-opcode VALU-rate and clock probe, an interleaved A/B, sustained runs with power logging, and a dump of the driver's ISA, which `gpu_isacheck.py` then audits here.
2. **State predictions as compile-based ranges.** G1 = 2585 VALU/hash (= cgminer), 2.0-2.5 GH/s. G2 = 2206, **1.10-1.17x**, 2.1-2.9 GH/s, and it needs ≥ ~4 waves/SIMD. Replace the README's "~3 GH/s" and "2,000-4,000x" with "~2-3 GH/s" and "2,000-6,000x".
3. **Measure energy or don't claim it.** Log board power and clock over sustained runs, add a power-cap sweep, and say that power tuning (+30% perf/W measured on an RX 9070) outweighs any kernel change here.
4. **Make it safe on a desktop GPU.** Use adaptive ~5-10 ms launches, cap G2's per-wave nonce loop, add a `--dedicated` mode, run periodic canary known-answer checks, and report failures as possible hardware errors.
5. **Cut and keep.** Cut G5, three of the four cgminer ports and the "3× `s_load_b512`" expectation. Keep G3's table: it beats inline (0.96x) and `v_readlane` (1.14x), and it lets G2 pay off from 32 versions.

Sources: RDNA whitepaper, quoted via search ("the instruction cache can deliver 32B (typically 2-4 instructions) every clock to each of the SIMDs"); [Chips and Cheese, RDNA 3 microbenchmarks](https://chipsandcheese.com/p/microbenchmarking-amds-rdna-3-graphics-architecture) (no INT32 VOPD; scalar-cache latency 15.4 ns); [Chips and Cheese, RDNA4 at Hot Chips 2025](https://chipsandcheese.com/p/amds-rdna4-gpu-architecture-at-hot) (~5-cycle vector latency); [Club386, RX 9070 at 171 W](https://www.club386.com/amd-radeon-rx-9070-maintains-93-of-its-performance-at-171w/); cgminer v3.7.2 `README`, `GPU-README`, `findnonce.c` and `driver-opencl.c`; tpruvot/ccminer `sha256/cuda_sha256d.cu`; chehw/spv-node `tests/opencl/hash256-asicboost.cl`.
