#!/usr/bin/env python3
"""Audit the GPU kernels as compiled for AMD RDNA4 (gfx1200, RX 9060 XT).

No GPU is needed: clang's AMDGPU backend compiles the same OpenCL source
that fbm-gpu hands the driver, and llvm-objdump disassembles it. For each
kernel's hot loop (one hash per iteration per lane) this counts what issues
per hash: VALU (vector ALU) instructions, which set the throughput of this
ALU-bound kernel, plus scalar ALU, scalar/vector memory and wait/delay
instructions. It checks that:

  - the loop's VALU count matches the generator's count (src/gen/stats.json)
    within a small budget: the compiler adds no work;
  - nothing spills to scratch and occupancy is the 16-wave maximum;
  - the version-lane loop gets its shared schedule through scalar loads (no
    per-lane vector loads), so every lane of a wave shares one copy;
  - each loop fits the 32 KB instruction cache.

Instructions inside branches that run only when a lane reports a candidate
(`s_cbranch_execz` skips them otherwise) are counted separately.

With bench/gpu-baselines/src/*.cl present (bench/gpu-baselines/fetch.sh),
cgminer 3.7.2's kernels are compiled and counted the same way.

Usage: gpu_isacheck.py [--mcpu gfx1200] [--binary FILE]
  --binary FILE  audit a program binary dumped by `fbm-gpu dump` (what the
                 installed driver actually compiled) instead of compiling
"""
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPU_CL = ['gpu/prelude.cl', 'src/gen/g_rdna_nonce.h', 'src/gen/g_rdna_vr.h', 'gpu/kernels.cl']
WG = 64
LANES = 2048  # RX 9060 XT: 32 CUs x 2 SIMD32
ICACHE = 32 * 1024
# Loop VALU beyond the generator's count: nonce byte swaps, the compare, loop
# control, and bfi/ch forms the ISA cannot encode (two literal operands).
BUDGET = 60

# cgminer 3.7.2 kernels, built as its host does for GCN: BITALIGN (media ops),
# no BFI_INT patching, one nonce per work-item.
BASELINES = ['poclbm130302.cl', 'phatk121016.cl', 'diakgcn121016.cl', 'diablo130302.cl']
SHIM = r'''
#define WORKSIZE 64
#define BITALIGN 1
#define get_local_id(d) __builtin_amdgcn_workitem_id_x()
#define get_group_id(d) __builtin_amdgcn_workgroup_id_x()
#define get_global_id(d) (__builtin_amdgcn_workgroup_id_x() * WORKSIZE + __builtin_amdgcn_workitem_id_x())
#define amd_bitalign(a, b, c) ((uint)(((((ulong)(a)) << 32) | (uint)(b)) >> ((uint)(c) & 31)))
#define amd_bytealign(a, b, c) ((uint)(((((ulong)(a)) << 32) | (uint)(b)) >> (((uint)(c) & 3) * 8)))
#define rotate(x, y) ((uint)(((x) << (y)) | ((x) >> (32 - (y)))))
#define bitselect(a, b, c) ((a) ^ ((c) & ((b) ^ (a))))
'''


def find_tools():
    for v in ('20', '19', '18', ''):
        clang = shutil.which('clang-' + v if v else 'clang')
        if not clang:
            continue
        objdump = (shutil.which('llvm-objdump-' + v) if v else None) or \
            ('/usr/lib/llvm-%s/bin/llvm-objdump' % v if v else shutil.which('llvm-objdump'))
        if objdump and os.path.exists(objdump):
            return clang, objdump
    sys.exit('need clang (with the AMDGPU target) and llvm-objdump, e.g. apt install clang-20 llvm-20')


def compile_cl(clang, objdump, text, mcpu, extra):
    with tempfile.TemporaryDirectory() as d:
        src, obj = os.path.join(d, 'k.cl'), os.path.join(d, 'k.o')
        with open(src, 'w') as f:
            f.write(text)
        r = subprocess.run([clang, '-x', 'cl', '-cl-std=CL1.2', '-target', 'amdgcn-amd-amdhsa',
                            '-mcpu=' + mcpu, '-O3', '-nogpulib', '-w',
                            '-Rpass-analysis=kernel-resource-usage', '-c', src, '-o', obj] + extra,
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit('compile failed:\n' + r.stderr[-4000:])
        asm = subprocess.run([objdump, '-d', '--mcpu=' + mcpu, obj], capture_output=True,
                             text=True, check=True).stdout
    return asm, resources(r.stderr)


def resources(remarks):
    """Per-kernel resource usage from clang's kernel-resource-usage remarks."""
    res, cur = {}, None
    for line in remarks.splitlines():
        m = re.search(r'remark: Function Name: (\w+)', line)
        if m:
            cur = res.setdefault(m.group(1), {})
            continue
        m = re.search(r'remark:\s+([\w ]+?)(?: \[[^\]]*\])?: (\w+) \[', line)
        if m and cur is not None:
            cur[m.group(1)] = m.group(2)
    return res


def parse(asm):
    funcs, cur, base = {}, None, 0
    for line in asm.splitlines():
        m = re.match(r'^([0-9a-f]+) <(\w+)>:', line)
        if m:
            cur, base = m.group(2), int(m.group(1), 16)
            funcs[cur] = []
            continue
        m = re.match(r'^\s+(\S+)\s*(.*?)\s*//\s*([0-9A-F]+):\s*([0-9A-F ]+?)\s*(?:<(\w+)\+0x([0-9a-f]+)>)?$',
                     line)
        if m and cur:
            addr = int(m.group(3), 16)
            size = len(m.group(4).split()) * 4
            tgt = int(m.group(6), 16) + base if m.group(6) else None
            funcs[cur].append((addr, m.group(1), size, tgt))
    return funcs


def classify(op):
    if op.startswith('v_dual'):
        return 'VALU'  # one VOPD issue = two ops; counted once, it is one slot
    if op.startswith('v_'):
        return 'VALU'
    if op.startswith(('s_load', 's_buffer_load')):
        return 'SMEM'
    if op.startswith(('s_wait', 's_delay_alu', 's_nop', 's_clause')):
        return 'wait'
    if op.startswith('s_'):
        return 'SALU'
    if op.startswith(('global_', 'buffer_', 'flat_', 'scratch_')):
        return 'VMEM'
    if op.startswith('ds_'):
        return 'LDS'
    return 'other'


def region_stats(ins, lo, hi):
    """Counts in [lo, hi], split into always-run and candidate-only parts."""
    # `s_and_saveexec; s_cbranch_execz X` skips to X when no lane reports a
    # candidate. Forward: [branch, X) is candidate-only. Backward (X is the
    # loop head, "next iteration"): everything after the branch is.
    cond = []
    for addr, op, size, tgt in ins:
        if lo <= addr <= hi and op == 's_cbranch_execz' and tgt is not None:
            cond.append((addr + size, tgt if tgt > addr else hi + 1))
    hot, rare = collections.Counter(), collections.Counter()
    ops, nbytes = collections.Counter(), 0
    for addr, op, size, tgt in ins:
        if not lo <= addr <= hi:
            continue
        nbytes += size
        c = classify(op)
        if any(a <= addr < b for a, b in cond):
            rare[c] += 1
        else:
            hot[c] += 1
            if c == 'VALU':
                ops[op] += 1
            if op.startswith('v_dual'):
                hot['VOPD'] += 1
    return hot, rare, ops, nbytes


def main_loop(ins):
    """The largest backward branch: the per-hash loop."""
    best = None
    for addr, op, size, tgt in ins:
        if tgt is not None and tgt < addr and op.startswith(('s_branch', 's_cbranch')):
            if best is None or addr - tgt > best[1] - best[0]:
                best = (tgt, addr)
    return best


def gen_counts():
    with open(os.path.join(ROOT, 'src/gen/stats.json')) as f:
        return {s['mode']: s for s in json.load(f) if s['target'] == 'rdna'}


def row(name, hot, rare, nbytes, res, gen=None):
    valu = hot['VALU']
    ghz = '%.2f-%.2f' % (LANES * 2.5 / valu, LANES * 3.1 / valu)
    print('| %s | %d | %s | %d | %d | %d | %d | %d | %d | %s | %s | %s | %s |' % (
        name, valu, '%+d' % (valu - gen) if gen is not None else '-', hot['SALU'], hot['SMEM'],
        hot['VMEM'], hot['wait'], rare['VALU'] + rare['SALU'] + rare['VMEM'], nbytes,
        res.get('VGPRs', '?'), res.get('ScratchSize', '?'), res.get('Occupancy', '?'), ghz))


def main():
    args = sys.argv[1:]
    mcpu = args[args.index('--mcpu') + 1] if '--mcpu' in args else 'gfx1200'
    clang, objdump = find_tools()
    if '--binary' in args:
        path = args[args.index('--binary') + 1]
        asm = subprocess.run([objdump, '-d', path], capture_output=True, text=True,
                             check=True).stdout
        res = {}
        print('driver-compiled binary %s' % path)
    else:
        text = ''.join(open(os.path.join(ROOT, p)).read() for p in GPU_CL)
        asm, res = compile_cl(clang, objdump, text, mcpu, ['-DFBM_AUDIT', '-DFBM_WG=%d' % WG])
        print('%s, -mcpu=%s, work-group %d' % (os.path.basename(clang), mcpu, WG))
    funcs = parse(asm)
    gen = gen_counts()

    print('\nper hash = per loop iteration of one lane; GH/s = %d lanes x 2.5-3.1 GHz / VALU\n'
          % LANES)
    print('| kernel | VALU/hash | vs generator | SALU | SMEM | VMEM | wait/delay | '
          'candidate-only | loop bytes | VGPRs | scratch | waves/SIMD | GH/s est. |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    ok = True
    for name, mode in (('fbm_nonce', 'nonce'), ('fbm_vr', 'vr')):
        # Driver binaries may decorate kernel symbols; match by substring.
        found = [f for f in funcs if f == name] or \
            [f for f in funcs if name in f and 'sched' not in f]
        if not found:
            print('missing kernel %s (symbols: %s)' % (name, ', '.join(funcs)))
            ok = False
            continue
        ins = funcs[found[0]]
        loop = main_loop(ins)
        if not loop:
            print('no loop found in %s' % found[0])
            ok = False
            continue
        lo, hi = loop
        hot, rare, ops, nbytes = region_stats(ins, lo, hi)
        g = gen[mode]['vector_instructions']
        r = res.get(name, {})
        row(name, hot, rare, nbytes, r, g)
        if hot['VALU'] - g > BUDGET or hot['VALU'] < g - 20:
            print('  FAIL: %s loop has %d VALU, generator says %d (budget +%d)'
                  % (name, hot['VALU'], g, BUDGET))
            ok = False
        if r and (r.get('ScratchSize') != '0' or r.get('Occupancy') != '16'):
            print('  FAIL: %s spills or runs below full occupancy' % name)
            ok = False
        if nbytes > ICACHE:
            print('  FAIL: %s loop is %d bytes, more than the %d-byte I$' % (name, nbytes, ICACHE))
            ok = False
        if mode == 'vr' and (hot['VMEM'] or hot['SMEM'] == 0):
            print('  FAIL: the vr loop must read its schedule with scalar loads only '
                  '(SMEM %d, VMEM %d)' % (hot['SMEM'], hot['VMEM']))
            ok = False
        top = ', '.join('%s %d' % kv for kv in ops.most_common(9))
        print('|  | %s |' % top + ' |' * 11)

    base_dir = os.path.join(ROOT, 'bench/gpu-baselines/src')
    if '--binary' not in args and os.path.isdir(base_dir):
        print('\nPrior art, cgminer 3.7.2 (one nonce per work-item, whole kernel):\n')
        print('| kernel | VALU/hash | vs generator | SALU | SMEM | VMEM | wait/delay | '
              'candidate-only | bytes | VGPRs | scratch | waves/SIMD | GH/s est. |')
        print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for b in BASELINES:
            p = os.path.join(base_dir, b)
            if not os.path.exists(p):
                continue
            basm, bres = compile_cl(clang, objdump, SHIM + open(p).read(), mcpu, [])
            ins = parse(basm)['search']
            hot, rare, ops, nbytes = region_stats(ins, ins[0][0], ins[-1][0])
            row(b.replace('.cl', ''), hot, rare, nbytes, bres.get('search', {}))
    print('\nok' if ok else '\nFAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
