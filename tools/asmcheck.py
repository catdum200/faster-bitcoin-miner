#!/usr/bin/env python3
"""Audit the compiled hot loops of the SIMD kernels.

For each scan function, finds the largest loop (backward branch) in the
object code and counts what executes per iteration:

  alu       vector ALU instructions (should match the generator's op count)
  gpr-bcast vpbroadcastd from a general-purpose register: an extra port-5
            uop on Skylake-SP; clang uses memory/embedded broadcasts instead
  reg-mov   register-to-register vector copies (only partly eliminated)
  spill     vector stores to memory (spills)

Exits 1 if a hot loop has more GPR broadcasts than the budget, or if its
ALU count differs from the generator's by more than 2%, which would mean
the compiler added work.

Usage: tools/asmcheck.py [build-dir]   (default: build)
"""
import json
import os
import re
import subprocess
import sys

GPR_BCAST_BUDGET = 2  # the per-iteration nonce-base broadcast is fine
ALU = re.compile(r'^(vpaddd|vpternlogd|vpro[lr]d|vpsrld|vpslld|vpxor[dq]?|vpand[dq]?|vpandn[dq]?|'
                 r'vpor[dq]?|vpshufb|vpcmpud|vpminud|vpcmpeqd|vmovmskps)$')
KERNELS = {
    # object file: [(function, generator target, generator mode)]
    'kernel_avx2.o': [('fbm_scan_avx2', 'avx2', 'nonce'), ('fbm_scan_avx2_vr', 'avx2', 'vr')],
    'kernel_avx512vl.o': [('fbm_scan_avx512vl', 'avx512', 'nonce'),
                          ('fbm_scan_avx512vl_vr', 'avx512', 'vr')],
    'kernel_avx512.o': [('fbm_scan_avx512', 'avx512', 'nonce'),
                        ('fbm_scan_avx512_vr', 'avx512', 'vr')],
}


def functions(obj):
    out = subprocess.run(['objdump', '-d', '--no-show-raw-insn', obj], capture_output=True,
                         text=True, check=True).stdout
    funcs = {}
    for chunk in re.split(r'\n(?=[0-9a-f]+ <)', out):
        m = re.match(r'[0-9a-f]+ <([^>]+)>:', chunk)
        if not m:
            continue
        ins = []
        for line in chunk.split('\n')[1:]:
            mm = re.match(r'\s*([0-9a-f]+):\s+(\S+)\s*(.*)', line)
            if mm:
                ins.append((int(mm.group(1), 16), mm.group(2), mm.group(3)))
        funcs[m.group(1)] = ins
    return funcs


def loops(ins):
    """All (start, end) index ranges closed by a backward branch."""
    index = {a: i for i, (a, _, _) in enumerate(ins)}
    for i, (a, op, args) in enumerate(ins):
        m = re.match(r'([0-9a-f]+)', args)
        if op.startswith('j') and m:
            target = int(m.group(1), 16)
            if target <= a and target in index:
                yield index[target], i


def with_callees(body, funcs, fn):
    """The compiler may outline the huge straight-line body: count local
    functions called from the loop as part of it."""
    out = list(body)
    for _, op, args in body:
        m = re.search(r'<([^>+]+)>', args)
        if op.startswith('call') and m and m.group(1) in funcs and m.group(1) != fn:
            out += funcs[m.group(1)]
    return out


def hot_loop(funcs, fn):
    """The loop that executes the most vector ALU work per iteration."""
    ins = funcs.get(fn, [])
    best, best_alu = [], -1
    for lo, hi in loops(ins):
        body = with_callees(ins[lo:hi + 1], funcs, fn)
        alu = sum(1 for _, op, _ in body if ALU.match(op))
        if alu > best_alu:
            best, best_alu = body, alu
    return best


def main():
    build = sys.argv[1] if len(sys.argv) > 1 else 'build'
    here = os.path.dirname(os.path.abspath(__file__))
    stats = {(s['target'], s['mode']): s for s in
             json.load(open(os.path.join(here, '..', 'src', 'gen', 'stats.json')))}
    ok = True
    print('%-22s %6s %6s %9s %9s %8s %6s' % ('hot loop', 'insns', 'alu', 'expected', 'gpr-bcast',
                                              'reg-mov', 'spill'))
    for obj, entries in KERNELS.items():
        path = os.path.join(build, obj)
        if not os.path.exists(path):
            continue
        funcs = functions(path)
        for fn, target, mode in entries:
            body = hot_loop(funcs, fn)
            alu = sum(1 for _, op, _ in body if ALU.match(op))
            gpr = sum(1 for _, op, a in body if op == 'vpbroadcastd' and re.match(r'%[er]', a))
            mov = sum(1 for _, op, a in body
                      if op.startswith('vmov') and re.fullmatch(r'%[xyz]mm\d+,%[xyz]mm\d+', a))
            spill = sum(1 for _, op, a in body
                        if op.startswith('vmov') and re.search(r'%[xyz]mm\d+,\S*\(', a))
            expected = stats[(target, mode)]['vector_instructions']
            bad = gpr > GPR_BCAST_BUDGET or abs(alu - expected) > 0.02 * expected
            ok &= not bad
            print('%-22s %6d %6d %9d %9d %8d %6d%s' % (fn, len(body), alu, expected, gpr, mov, spill,
                                                     '  <-- FAIL' if bad else ''))
    print('ok' if ok else 'FAILED: hot-loop codegen regressed (see columns above)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
