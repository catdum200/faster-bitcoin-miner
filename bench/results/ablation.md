
**AVX-512 (16 lanes)**, vector instructions per hash:

| step | per hash | vs previous | total reduction |
|---|---:|---:|---:|
| naive: 3 compressions per nonce, nothing precomputed | 298.5 | - | - |
| + midstate (T1): 2 compressions per nonce | 197.7 | -33.8% | 1.51x |
| + constant folding / precompute (T2, T3) | 185.4 | -6.2% | 1.61x |
| + early exit after round 60 of hash 2 (T4) | 177.9 | -4.0% | 1.68x |
| + version rolling, 128 versions per nonce (T7) | 149.2 (+4.2 scalar per hash for the shared schedule) | -16.2% | 2.00x |

**AVX2 (8 lanes)**, vector instructions per hash:

| step | per hash | vs previous | total reduction |
|---|---:|---:|---:|
| naive: 3 compressions per nonce, nothing precomputed | 1196.2 | - | - |
| + midstate (T1): 2 compressions per nonce | 793.0 | -33.7% | 1.51x |
| + constant folding / precompute (T2, T3) | 748.5 | -5.6% | 1.60x |
| + early exit after round 60 of hash 2 (T4) | 718.6 | -4.0% | 1.66x |
| + version rolling, 128 versions per nonce (T7) | 613.6 (+4.2 scalar per hash for the shared schedule) | -14.6% | 1.95x |
