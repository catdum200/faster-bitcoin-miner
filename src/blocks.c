/* Real mainnet block headers with known answers, shared by the CPU and GPU
 * test suites. */
#include "blocks.h"

/* Real mainnet headers: genesis (height 0) and height 125552. */
#define GENESIS_HEX \
    "0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27a" \
    "c72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d1dac2b7c"
#define BLOCK125552_HEX \
    "0100000081cd02ab7e569e8bcd9317e2fe99f2de44d49ab2b8851ba4a308000000000000e320b6c2fffc8d75" \
    "0423db8b1eb942ae710e951ed797f7affc8892b0f1fc122bc7f5d74df2b9441a42a14695"
const char *const fbm_genesis_hex = GENESIS_HEX;
const char *const fbm_block125552_hex = BLOCK125552_HEX;

/* Mainnet blocks whose versions have BIP 320 bits rolled by the ASIC that
 * mined them: heights 800000-910000 and three from September 2026 (ViaBTC,
 * Braiins Pool, MARA Pool). Hashes were checked against mempool.space. */
const fbm_known_block fbm_known_blocks[] = {
    {"genesis", GENESIS_HEX},
    {"125552", BLOCK125552_HEX},
    {"800000",
     "00601d3455bb9fbd966b3ea2dc42d0c22722e4c0c1729fad172101000000000000000000550"
     "87fab0c8f3f89f8bcfd4df26c504d81b0a88e04907161838c0c53001af09135edbd64943805175e955e06"},
    {"850000",
     "0080bb2b13b3152752d9cf2a36fcd16d78f64269d847932f076b02000000000000000000d51"
     "a6bd669cf6bc30269a259c2918e6319269fc15f020a82ddd6a5fdc1f5cf71ca618066255d03176ab0fb7c"},
    {"900000",
     "00a0ab20247d4d9f582f9750344cdf62c46d81d046be960340960100000000000000000070f"
     "96945530651135839d8adc3f40e595118ec74c7ad81a3d17bb022e554fb0c937f4268743702177ad05f92"},
    {"910000",
     "00a0572be06d4f01a2ed2228dec965539cc8b96512ccde7d2824010000000000000000006f2"
     "8c30dc748f6b1430fb2b9a5a94b5b34a5df6e318c6cc5c310a1a35b432b59a3ab9d68b32c021719d103e9"},
    {"968566",
     "0000003874fd1f0f0ea2296a90502e569622d8e3bc1f8d201d650000000000000000000001c20b8594f8"
     "5b5e3d92e4a66ac4f2864913fcd8b44c4a3bbf7da256378081ec2fa1b66ac51e02177b22a039"},
    {"968562",
     "0060aa297152dfd5bda8f97830a47df977bf41a6254159cc0366000000000000000000008abded7270cd"
     "1dfde5a3cfd286b8a1787980c006b9f35b8f47146dc68abdbac0ef95b66ac51e021718227e7a"},
    {"968555",
     "00607925ad9b791fcca5e6d220cd922225cc79e7811a78d775110000000000000000000033382e41f38a"
     "d364b21b717a3baa7cdf34e834c52bbcd3f800d785aeac3539ff6a7eb66ac51e021748202491"},
};
const size_t fbm_known_block_count = sizeof fbm_known_blocks / sizeof fbm_known_blocks[0];
