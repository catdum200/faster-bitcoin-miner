#!/bin/sh
# Fetch cpuminer-opt (JayDDee, GPL-2) at a pinned release for the optional
# baseline kernels. The source is not part of this repository.
set -e
dir=$(cd "$(dirname "$0")" && pwd)/src
commit=b34565bfaca5da9ed894a326e3ba0dfe459550a4 # v26.1
if [ ! -d "$dir/.git" ]; then
    git clone --quiet https://github.com/JayDDee/cpuminer-opt "$dir"
fi
git -C "$dir" -c advice.detachedHead=false checkout --quiet "$commit"
touch "$dir/cpuminer-config.h"
# clang rejects a non-constant immediate in this inline helper (gcc only
# accepts it after inlining). Turning it into the equivalent macro lets the
# baseline be built with clang, its fastest compiler here (+17% vs gcc).
# The SHA-256 code itself is untouched.
sed -i 's/^static inline __m512i mm512_perm128( const __m512i v, const int c )$/#define mm512_perm128( v, c ) _mm512_shuffle_i64x2( v, v, c )/; /^{  return _mm512_shuffle_i64x2( v, v, c ); }$/d' \
    "$dir/simd-utils/simd-512.h"
echo "cpuminer-opt $(git -C "$dir" describe --tags) ($commit) in $dir"
