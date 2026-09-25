CC ?= cc
CFLAGS ?= -O3 -g
CFLAGS += -std=gnu11 -Wall -Wextra -pthread
LDLIBS = -lcrypto -pthread

SRC = main.c selftest.c miner.c kernels.c sha256.c header.c \
      kernel_ref.c kernel_scalar.c kernel_avx2.c kernel_avx512.c kernel_avx512vl.c
OBJ = $(SRC:%.c=build/%.o)
GEN = $(wildcard src/gen/*.h)

# Only the SIMD kernels get ISA flags; they are selected at run time after
# a CPUID check, so the binary still runs on any x86-64 CPU.
build/kernel_avx2.o: ISA = -mavx2
build/kernel_avx512.o build/kernel_avx512vl.o: ISA = -mavx512f -mavx512bw -mavx512vl

.PHONY: all gen test asan clean

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

# Tests under AddressSanitizer + UndefinedBehaviorSanitizer.
asan:
	$(MAKE) clean
	$(MAKE) CFLAGS="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" fbm
	./fbm test
	$(MAKE) clean

clean:
	rm -rf build fbm
