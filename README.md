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
   where the critique was right. For example, the version-rolling gain
   vanishes unless 64 or more versions share each nonce's schedule.

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
python3 tools/economics.py --mhs 128 --watts 30    # live network numbers
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
tools/asmcheck.py      audits the compiled hot loops (make check)
tools/economics.py     revenue/energy from live mempool.space data
bench/                 baseline fetch/build scripts, run_all.sh, results/
docs/                  PLAN_v1 -> CRITIQUE -> PLAN -> RESULTS
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
