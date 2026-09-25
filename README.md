# faster-bitcoin-miner

A SHA-256d (Bitcoin proof-of-work) search engine for x86-64 CPUs, built to
answer one question: **is there a more efficient way to mine bitcoin?**

## The honest answer

1. **There is no algorithmic shortcut.** Mining is a brute-force search for a
   header whose `SHA256(SHA256(header))` is below a target. Predicting good
   nonces (ML, patterns, "quantum") would require breaking SHA-256. What
   software *can* do is skip work that the header's layout makes redundant.
2. **This project implements every such trick we know of, plus one that
   existing CPU miners lack.** In SIMD lanes it shares the message schedule
   across BIP 320 rolled versions: "overt AsicBoost" (2016), applied to
   AVX-512. On this 4-core Cascade Lake VM it reaches **124.7 MH/s**, which is
   **1.21x the fastest open-source CPU sha256d code we know** (cpuminer-opt's
   16-way AVX-512 kernel: 1.207x, 95% CI [1.194, 1.223], same machine and
   harness) and **35x a naive implementation**. The prior-art tricks alone
   only reach parity (1.03x); version rolling is the whole gain.
3. **None of this makes CPU mining worthwhile.** At 124.7 MH/s this machine
   would earn about **$0.002 per year**, while its electricity alone costs
   about $26. Solo, it would find a block about once every **145 million
   years**. Per hash, it uses about **18,000x more energy** than a current
   ASIC. In the real world, mining efficiency comes from ASIC silicon
   (~10-15 J/TH), cheap electricity and pools. The version-rolling trick used
   here already ships in ASICs: 14 of the 15 most recent blocks we checked
   have rolled version bits.

The full numbers and the method are in [`docs/RESULTS.md`](docs/RESULTS.md).

## What about a home GPU? (e.g. Radeon RX 9060 XT 16 GB)

Still a bad idea for bitcoin. But this repository now has a GPU miner for
that card: `fbm-gpu` (OpenCL). Its kernel does **1.15-1.17x less work per
hash than the best open-source GPU kernels** (cgminer, 2013), by rolling
BIP 320 versions across GPU lanes. See [`docs/GPU_RESULTS.md`](docs/GPU_RESULTS.md).

- **The gain is static.** No GPU was available: the kernels were compiled for
  the card's ISA (gfx1200) and every instruction in the hot loop was counted.
  The same was done for cgminer's kernels. Nothing was timed on the card.
- **What was tested.** The kernels and host pass the correctness suite on a
  CPU OpenCL (PoCL). The Windows build passes it under Wine.
- **Hash rate.** ~2-3 GH/s. This is a compile-based estimate, unmeasured:
  2.1-2.9 GH/s for the version-rolling kernel, 1.8-2.5 for prior art, at the
  card's 2.5-3.1 GHz. A published hashcat run fits it: 12.78 GH/s SHA-1 and
  1.40 GH/s SHA-512 on this card suggest ~5 GH/s of single SHA-256, and mining
  costs almost two SHA-256 compressions per attempt.
- **Money.** At 2.3 GH/s, `python3 tools/economics.py --mhs 2300 --watts 160
  --usd-per-kwh 0.15 --rent-usd-per-hour 0` gives **$0.034 of expected revenue
  per year** against **$210 per year** of electricity for the card alone.
- **Solo odds.** About one block per **8 million years**.
- **Energy.** **2,000-6,000x worse per hash** than an Antminer S21 XP
  (13.5 J/TH).
- **Better code does not change this.** The 1.17x kernel moves revenue from
  about three cents a year to about four. Lowering the power limit is a
  bigger efficiency lever than any kernel, and it does not change the
  verdict either.

**To measure it on your card:** on Windows run
`dist\fbm-gpu.exe report` (see [`dist/README.md`](dist/README.md)); on
Linux, `make gpu && ./fbm-gpu report`. It writes one file with:

- the tests on the real GPU;
- per-instruction issue rates;
- an interleaved speed comparison against cgminer's kernel;
- sustained runs, logging power on Linux;
- the driver-compiled binary.

Sources: [RX 9060 XT hashcat results (OpenBenchmarking)](https://openbenchmarking.org/result/2506064-PTS-NEWGPUCO48),
[RX 6800 XT hashcat benchmarks](https://gist.github.com/epixoip/99085955a1145ff61ec83512a50421a7),
[RX 9060 XT specs, Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/amd-radeon-rx-9060-xt-16gb-review/7),
[RDNA4 ISA guide (AMD)](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf).

## How it was built: plan, critique, revise

1. [`docs/PLAN_v1.md`](docs/PLAN_v1.md): the first plan.
2. [`docs/CRITIQUE.md`](docs/CRITIQUE.md): an independent agent's adversarial
   review, with its own measurements on this machine. Its main points:
   - the baseline was a strawman;
   - most of the "techniques" were 2011-2016 prior art;
   - TSC cycles are not core cycles;
   - the statistics were too weak;
   - the scalar schedule in the version-rolling design was not free.
3. [`docs/PLAN.md`](docs/PLAN.md): the revised plan. It maps every finding to
   a change, and the code follows it.
4. [`docs/RESULTS.md`](docs/RESULTS.md): the measured outcome, including
   where the critique was right. For example, version rolling is a net
   *loss* when only 16 versions share each nonce's schedule, and needs about
   128 to reach its full +15%.

The GPU kernel went through the same process:
[`GPU_PLAN_v1`](docs/GPU_PLAN_v1.md) → [`GPU_CRITIQUE`](docs/GPU_CRITIQUE.md)
→ [`GPU_PLAN`](docs/GPU_PLAN.md) → [`GPU_RESULTS`](docs/GPU_RESULTS.md).

## How it works

`gen/gen_kernels.py` is a small **partial evaluator**. It classifies every
intermediate value of the double SHA-256 by how often it changes:

- constant
- per job
- per SIMD lane
- per nonce
- per lane-and-nonce

It then folds constants, hoists everything that isn't per-lane-and-nonce into
scalar precompute code, and emits fully unrolled SIMD code only for the rest.
It also removes the second hash's last rounds: only H7 has to be zero, and it
is known after round 60. The same generator produces the scalar, AVX2 and
AVX-512 kernels, each in two layouts:

| layout | lanes hold | block-2 message schedule |
|--------|------------|--------------------------|
| nonce (`avx512`, `avx2`, ...) | 16 or 8 nonces, same version | computed per lane (vector work) |
| version (`avx512-vr`, ...) | 16 or 8 rolled versions, same nonce | identical in every lane: computed once per nonce in scalar code and shared by 128 versions |

| technique | where it comes from |
|-----------|---------------------|
| Midstate: the first 64 header bytes are hashed once per job | prior art (early GPU/CPU miners, ~2011) |
| Precomputing rounds 0-2 and constant schedule words; folding zero padding words | prior art (pooler/cpuminer, phatk, ~2011-2013) |
| Early exit after round 60 of hash 2 (H7 only) | prior art (phatk and cpuminer, ~2011) |
| 16-way AVX-512 using `vprord` and `vpternlogd` | prior art (cpuminer-opt) |
| Sharing the block-2 schedule across rolled versions (AsicBoost / BIP 320) | ASICs since 2016-2018; **new here relative to the CPU miners compared** |
| Emitting the next round's `h+K+W` before `Ch`, so `vpternlogd` overwrites a dead register (~100 fewer register copies per 16 hashes) | this project |

## Usage

```sh
make                 # builds ./fbm (prefers clang; about 20% faster AVX2 than gcc 13)
make check           # correctness tests + compiled hot-loop audit
make asan tsan       # the tests under Address/UB/Thread sanitizers
./fbm list           # kernels and what this CPU supports
./fbm test           # correctness tests only
./fbm freq           # measured core clock for scalar / 256-bit / 512-bit code
./fbm bench --threads 4 --rounds 20
./fbm mine --header <160 hex chars> [--start N --count N --versions N --threads N]
python3 tools/economics.py --mhs 124.7 --watts 30  # live network numbers

make gpu             # builds ./fbm-gpu (OpenCL, loaded at run time; no SDK needed)
./fbm-gpu list       # OpenCL devices
./fbm-gpu test       # GPU kernels vs the reference (any OpenCL device, PoCL included)
./fbm-gpu bench      # interleaved: cgminer's poclbm vs nonce-lane vs version-lane kernel
./fbm-gpu mine --header <160 hex> --versions 65536 [--seconds S]
./fbm-gpu report     # everything, in one file to send back (run on the real card)
make gpu-check       # gfx1200 ISA audit (needs clang >= 18 with AMDGPU)
make gpu-win         # Windows build (mingw-w64); prebuilt: dist/fbm-gpu.exe
```

`make BASELINES=1` also fetches [cpuminer-opt](https://github.com/JayDDee/cpuminer-opt)
v26.1 (GPL-2; not included in this repo) and links its sha256d kernels into
`fbm` as the kernels `cpuminer-opt16` and `cpuminer-opt8`. They are
benchmarked under the same harness, and they pass the same correctness
tests. `bench/cpuminer/build.sh` builds a harness for pooler/cpuminer's AVX2
assembly. `bench/run_all.sh` reproduces every number in the results.

Example: re-mining block 125552 on 4 threads prints its real nonce:

```sh
./fbm mine --header 0100000081cd02ab7e569e8bcd9317e2fe99f2de44d49ab2b8851ba4a308000000000000e320b6c2fffc8d750423db8b1eb942ae710e951ed797f7affc8892b0f1fc122bc7f5d74df2b9441a00000000 \
    --start 2400000000 --count 200000000
# SOLUTION version=0x00000001 nonce=2504433986 hash=00000000000000001e8d6829a8a21adc5d38d0a473b144b6765798e61f98bd1d
```

## Correctness

The portable reference SHA-256 is the oracle. It is checked against the FIPS
180 vectors and against OpenSSL for every length from 0 to 299. Every kernel,
the cpuminer-opt baselines included, must report *exactly* the reference's
candidate set in these cases:

- random headers, with loose and tight thresholds;
- odd sizes;
- the end of the nonce space;
- thread splits of 1, 4 and 7.

Every kernel must also rediscover the real (version, nonce) of 9 mainnet
blocks. These are genesis, block 125552, and 7 blocks mined with rolled
versions (heights 800000 to 968566). For those 7, the job starts from a
different version, so the kernel must find the real version by rolling.

`fbm mine` treats a candidate that fails its own filter on re-hashing as a
kernel bug and aborts, instead of silently discarding it.

## Layout

```
gen/gen_kernels.py     partial evaluator -> src/gen/*.h (committed; `make gen` regenerates)
src/kern_template.h    the two kernel layouts around the generated code
src/kernel_*.c         per-ISA instantiations; ref/OpenSSL baselines; cpuminer-opt wrapper
src/miner.c            multithreaded driver (pinned threads, contiguous nonce slices)
src/main.c             CLI: test, list, freq, bench (interleaved, bootstrap CIs), mine
gpu/                   OpenCL kernels (around the generated rdna code) and issue-rate probes
src/gpu*.c, clload.c   fbm-gpu: OpenCL driver, tests, bench, mine, report
tools/gpu_isacheck.py  audits the kernels compiled for gfx1200 (make gpu-check)
dist/fbm-gpu.exe       prebuilt Windows fbm-gpu
tools/asmcheck.py      audits the compiled hot loops (make check)
tools/economics.py     revenue/energy from live mempool.space data
bench/                 baseline fetch/build scripts, run_all.sh, results/
docs/                  PLAN_v1 -> CRITIQUE -> PLAN -> RESULTS, and the same for GPU_*
```

## Limitations and next steps

- **AVX2-only CPUs.** The unrolled 256-bit kernels are front-end-bound:
  16-30 KB loops run from the legacy decoders. A partially rolled kernel that
  fits the micro-op cache is the most promising next step for CPUs without
  AVX-512. Relatedly, version rolling did not help the 256-bit AVX-512VL
  kernel; see "Open issues" in the results.
- **SHA-NI.** Most current CPUs (AMD Zen, Intel Ice Lake and later) have SHA
  extensions, which would beat AVX2 there. This VM has no SHA-NI, so a kernel
  could not be tested and none is included.
- **Stratum.** Pool networking is not implemented. Version rolling in a pool
  also needs BIP 310 negotiation.
- **Measured on one machine.** All numbers are from one 4-core Cascade Lake
  VM. AVX-512 drops the clock (about 3.3 GHz scalar vs 2.7 GHz with zmm), and
  another CPU will differ.
