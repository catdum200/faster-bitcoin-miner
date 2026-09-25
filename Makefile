# clang schedules the huge unrolled AVX2 kernels better (about 20% faster
# than gcc 13 on Cascade Lake); AVX-512 speed is the same with either.
ifeq ($(origin CC),default)
CC := $(shell command -v clang >/dev/null 2>&1 && echo clang || echo cc)
endif
CFLAGS ?= -O2 -g
.DEFAULT_GOAL := all
CFLAGS += -std=gnu11 -Wall -Wextra -pthread
LDLIBS = -lcrypto -pthread

SRC = main.c selftest.c miner.c kernels.c sha256.c header.c freq.c \
      kernel_ref.c kernel_scalar.c kernel_avx2.c kernel_avx512.c kernel_avx512vl.c
OBJ = $(SRC:%.c=build/%.o)
GEN = $(wildcard src/gen/*.h)

# Only the SIMD kernels get ISA flags; they are selected at run time after
# a CPUID check, so the binary still runs on any x86-64 CPU.
build/kernel_avx2.o: ISA = -mavx2
build/kernel_avx512.o build/kernel_avx512vl.o: ISA = -mavx512f -mavx512bw -mavx512vl

# Optional prior-art baseline kernels: `make BASELINES=1` fetches cpuminer-opt
# (GPL-2) at a pinned release and links its sha256d kernels into fbm, built
# the way its own build.sh does (-O3 -march=native).
ifdef BASELINES
CMO = bench/cpuminer-opt/src
OBJ += build/kernel_cpuminer_opt.o build/cmo_sha256_4way.o build/cmo_simd_constants.o
build/kernels.o: CFLAGS += -DFBM_BASELINES
build/kernel_cpuminer_opt.o: ISA = -mavx512f -mavx512bw -mavx512vl
$(CMO)/cpuminer-config.h:
	bench/cpuminer-opt/fetch.sh
build/cmo_sha256_4way.o: $(CMO)/cpuminer-config.h | build
	$(CC) -O3 -march=native -I$(CMO) -I$(CMO)/algo/sha -c -o $@ $(CMO)/algo/sha/sha256-hash-4way.c
build/cmo_simd_constants.o: $(CMO)/cpuminer-config.h | build
	$(CC) -O3 -march=native -I$(CMO) -c -o $@ $(CMO)/simd-utils/simd-constants.c
endif

.PHONY: all gen test check asan tsan clean

all: fbm

fbm: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

build/%.o: src/%.c $(wildcard src/*.h) $(GEN) | build
	$(CC) $(CFLAGS) $(ISA) -c -o $@ $<

build:
	mkdir -p build

# Regenerate the unrolled kernels (committed, so Python is optional).
gen:
	python3 gen/gen_kernels.py src/gen

test: fbm
	./fbm test

# Tests plus the hot-loop codegen audit (fails on GPR broadcasts or extra
# ALU work in the compiled SIMD loops).
check: fbm
	./fbm test
	python3 tools/asmcheck.py build

# Tests under AddressSanitizer + UndefinedBehaviorSanitizer (gcc: clang's
# sanitizer runtimes are often not installed). ASan's fake stack is off
# because GCC 13 misaligns 64-byte-aligned locals on it (GCC PR 110027,
# fixed in GCC 14), which faults on aligned AVX-512 stores.
asan:
	$(MAKE) clean
	$(MAKE) CC=gcc CFLAGS="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" fbm
	ASAN_OPTIONS=detect_stack_use_after_return=0 UBSAN_OPTIONS=halt_on_error=1 ./fbm test
	$(MAKE) clean

# Tests under ThreadSanitizer (the multithreaded driver).
tsan:
	$(MAKE) clean
	$(MAKE) CC=gcc CFLAGS="-O1 -g -fsanitize=thread" fbm
	./fbm test
	$(MAKE) clean

clean:
	rm -rf build fbm
