# GPU plan v2: revised after the critique

History:

- `GPU_PLAN_v1.md` was the first draft.
- `GPU_CRITIQUE.md` is an independent agent's adversarial review of it,
  with its own compiled prototypes and a PoCL run.
- This file is the plan the code follows.

As on the CPU, implementation started while the review was running: the
generator target, both kernels, the host, and the PoCL tests. So the
critic's own measurements (G1 2585, G2 2206 VALU/hash) and the committed
code (2580, 2196) agree to within a few instructions, and some findings were
already handled. The table says which.

## 1. What the critique changed

| # | Finding | Response |
|---|---------|----------|
| 1 | **[critical]** The counts are solid, but nothing turns them into GH/s or J/TH: the full-rate integer issue, the sustained clock and the power are all unmeasured, and the author cannot iterate on the card | **Accepted.** Add **`fbm-gpu report`**: one portable command that writes one file for the owner to send back. It contains: device, driver and OpenCL versions; the full test suite; a per-opcode **VALU rate probe** (the GPU analogue of `fbm freq`); the interleaved A/B bench; a sustained run that logs board power and clock where the OS exposes them (Linux hwmon); and a dump plus self-audit of the driver-compiled binary. It replaces the shell script `bench/gpu_run_all.sh`, which Windows users could not run. The owner can also run `hashcat -b -m 1400` as an independent calibration |
| 2 | **[major]** G2's gain is 1.10-1.17x, not 1.15-1.18x: its 261 non-VALU instructions per hash (mostly `s_delay_alu`) may take issue slots, and it has less ILP, so it needs occupancy | **Accepted.** Quote **1.10-1.17x** until measured. The audit now reports every instruction class, plus a pessimistic "every instruction is an issue slot" ratio. It fails on scratch or on fewer than 4 waves/SIMD, including for the driver's own binary. G2 still runs at the 16-wave maximum (67 VGPRs) |
| 3 | **[major]** Energy efficiency is claimed, never measured; the power limit (+32% perf/W on an RX 9070) outweighs any kernel change | **Accepted.** No J/TH claim without logged power. `fbm-gpu report` and `fbm-gpu sustain` log board power on Linux (`power1_average`, `freq1_input`); on Windows they say to read it from Adrenalin or HWiNFO. The docs give the power-cap sweep, and state plainly that power tuning is the bigger lever and that neither changes the economics |
| 4 | **[major]** The audit checks clang 20's output, not the driver's | **Accepted.** `fbm-gpu dump` saves `CL_PROGRAM_BINARIES`, and `gpu_isacheck.py --binary` audits it: VALU, loop shape, VGPRs, scratch and wave size from the ELF notes. The tolerance is ±2%, not equality. `fbm-gpu --program FILE` runs a clang-built code object, so a run that passes the tests executes exactly the audited ISA, where the runtime accepts it |
| 5 | **[major]** 100 ms launches freeze a desktop, and G2 waves live for the whole launch | **Accepted.** Launches are **adaptive, 8 ms by default**: the host measures the rate and sizes the next launch. `--dedicated` means 100 ms. G2's per-wave nonce loop is capped at 64. `mine` gets `--seconds` and a clean stop on Ctrl-C |
| 6 | **[major]** The README's ~3 GH/s is above the plan's own model | **Accepted.** From compiled counts: nonce layout 2.0-2.5 GH/s, version layout 2.1-2.9 GH/s. The README will say **"~2-3 GH/s (compile-based estimate, unmeasured)"** and **"2,000-6,000x worse per hash than an ASIC"** |
| 7 | **[major]** G1 duplicates the cgminer kernels, and porting them costs more than the plan says (GPL-3 host precompute, H7-only compare, racy hit writes) | **Accepted.** G1 stays as the same-harness nonce-lane reference. All four cgminer kernels stay as static counts. **One** is ported to run on the card: poclbm (public domain), with a clean-room precompute derived from the SHA-256 algebra, checked on the known blocks with their true versions. It is fetched, never vendored |
| 8 | **[minor]** G5 (explicit 3-input add trees) adds nothing | **Right about speed.** Binary trees compile to 2578/2194 VALU and ternary to 2580/2196, 0.08% worse. **Kept only as a cost model:** with ternary trees the generator's count predicts the compiled loop within 1%, which the audit checks; with binary trees it is 15% off. The claim moves from "technique" to "model" |
| 9 | **[minor]** G3's reasoning is partly wrong, though its choice is right; G2 pays off from 32 versions | **Accepted.** New rationale: the SALU has no 32-bit rotate, portable OpenCL puts wave-uniform rotates on the VALU (measured 0.96x of G1), and `v_readlane` costs VALU and occupancy (1.14x). The audit checks "no VMEM in the loop except the hit path, SMEM ≤ 8", not a load shape. The pre-pass is ~0.5 VALU per hash even for one wave of versions |
| 10 | **[major]** PoCL tests neither the card's compiler nor its hardware | **Accepted.** On-card tests add **planted answers at production geometry**: a real block whose answer sits at the highest version index and the last nonce of a multi-launch scan. `mine` runs a periodic **canary**: a loose-threshold scan checked against the host reference. A failed canary or re-check stops with "possible hardware error (clocks/undervolt?)" |
| 11 | **[minor]** "Builds on Windows" is untested and asks a lot of a home user | **Accepted.** The host loads OpenCL at run time (`OpenCL.dll` / `libOpenCL.so.1`), so it needs no SDK or import library. `make gpu-win` cross-compiles `fbm-gpu.exe` with Ubuntu's mingw-w64, and it is tested here under Wine against PoCL if possible |
| 12 | **[minor]** Fact-table details: VOP3+literal instructions, I$, wave64, "no memory traffic" | **Accepted.** The audit counts 96-bit (VOP3 + literal) instructions. The docs say the loops fit the I$, that wave64 is not a throughput risk, and that G2 reads 192 bytes per nonce per wave from the table |

## 2. Goal, restated honestly

Build the most efficient SHA-256d kernel we can for the RX 9060 XT
(gfx1200), with:

- exact static evidence (compiled instruction counts) of where it stands
  against the best existing GPU code;
- one command that lets the card's owner measure speed, correctness and
  power on the real hardware.

The one new technique is version rolling across lanes (G2), with an honest
range of **1.10-1.17x** over prior art until measured. Say plainly that:

- mining on this card loses money at any kernel speed (~$0.03-0.04 of
  revenue per year against ~$120-210 of electricity);
- the power limit is a bigger efficiency lever than any kernel change.

## 3. Work items (in order)

1. **Audit (`tools/gpu_isacheck.py`):**
   - check the generator count within ±2%;
   - report every instruction class and the 96-bit count, with a pessimistic
     issue-slot ratio;
   - fail on scratch, < 4 waves, or VMEM/SMEM in the vr loop;
   - in `--binary` mode, parse the ELF notes.
2. **Desktop safety:**
   - adaptive 8 ms launches, `--dedicated`, G2 loop cap;
   - `mine --seconds`, Ctrl-C;
   - the canary.
3. **On-card tests:** planted answers at production geometry.
4. **Probe:** per-opcode VALU rate kernels; their compiled loops are checked by
   the audit.
5. **Report:** `fbm-gpu report` (and `sustain` with power logging); remove
   `bench/gpu_run_all.sh`.
6. **Baseline:** poclbm on the card with a clean-room precompute.
7. **Windows:** run-time OpenCL loading, `make gpu-win`, a Wine+PoCL smoke
   test.
8. **Optional:** `--program FILE` (run a clang-built code object).
9. **Docs:** `GPU_RESULTS.md`, and README, CLAUDE.md numbers from the
   compiled counts.

## 4. Non-goals

- Claiming a measured GH/s, J/TH or speedup for the RX 9060 XT. None can be
  measured from this VM. The report command exists so that the owner can.
- A Stratum client; pools would also need BIP 310 for version rolling.
- HIP/CUDA/Vulkan ports. ccminer's CUDA sha256d (NVIDIA-only) is noted as
  prior art.
