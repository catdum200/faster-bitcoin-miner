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
# Loop VALU vs the generator's count: the compiler adds byte swaps, the
# compare and loop control, splits a few bfi into and+xor, and fuses a few
# xor+add into v_xad_u32. Anything beyond 2% means codegen went wrong.
TOLERANCE = 0.02

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


def find_tools(need_clang=True):
    if not need_clang:
        for c in ('llvm-objdump', 'llvm-objdump-20', 'llvm-objdump-19', 'llvm-objdump-18'):
            if shutil.which(c):
                return None, shutil.which(c)
        for v in ('20', '19', '18'):
            if os.path.exists('/usr/lib/llvm-%s/bin/llvm-objdump' % v):
                return None, '/usr/lib/llvm-%s/bin/llvm-objdump' % v
        sys.exit('need llvm-objdump (LLVM 18 or newer) to disassemble the binary')
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
                if size >= 12:  # VOP3 + 32-bit literal: a 96-bit instruction
                    hot['VALU96'] += 1
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


def notes_resources(path):
    """Per-kernel resources from a code object's metadata notes (for driver
    binaries, where there are no compiler remarks). Occupancy is estimated
    with LLVM's gfx11/gfx12 model: 1536 VGPRs per SIMD (768 for wave64) in
    granules of 24 (12)."""
    readelf = shutil.which('llvm-readelf') or next(
        (p for p in ('/usr/lib/llvm-%s/bin/llvm-readelf' % v for v in ('20', '19', '18'))
         if os.path.exists(p)), None)
    if not readelf:
        return {}
    text = subprocess.run([readelf, '--notes', path], capture_output=True, text=True).stdout
    res, cur = {}, None
    for line in text.splitlines():
        m = re.match(r'\s*(?:- )?\.(\w+):\s+(\S+)', line)
        if not m:
            continue
        key, val = m.groups()
        if key == 'name':
            cur = res.setdefault(val, {})
        elif cur is not None and key in ('vgpr_count', 'sgpr_count', 'private_segment_fixed_size',
                                         'wavefront_size'):
            cur[key] = int(val)
    out = {}
    for name, r in res.items():
        if 'vgpr_count' not in r:
            continue
        w64 = r.get('wavefront_size') == 64
        total, gran = (768, 12) if w64 else (1536, 24)
        v = max(1, r['vgpr_count'])
        out[name] = {'VGPRs': str(v), 'ScratchSize': str(r.get('private_segment_fixed_size', '?')),
                     'Occupancy': str(min(16, total // (-(-v // gran) * gran))),
                     'wave': str(r.get('wavefront_size', '?'))}
    return out


def issue_slots(hot):
    """Pessimistic: every instruction class takes the SIMD's issue slot."""
    return hot['VALU'] + hot['SALU'] + hot['SMEM'] + hot['VMEM'] + hot['wait'] + hot['other']


HEADER = ('| kernel | VALU/hash | vs generator | 96-bit VALU | SALU | SMEM | VMEM | wait/delay | '
          'all instr. | candidate-only | bytes | VGPRs | scratch | waves/SIMD | '
          'GH/s (VALU) | GH/s (all) |\n'
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')


def row(name, hot, rare, nbytes, res, gen=None):
    valu, slots = hot['VALU'], issue_slots(hot)
    print('| %s | %d | %s | %d | %d | %d | %d | %d | %d | %d | %d | %s | %s | %s | %s | %s |' % (
        name, valu, '%+.1f%%' % (100.0 * (valu - gen) / gen) if gen else '-', hot['VALU96'],
        hot['SALU'], hot['SMEM'], hot['VMEM'], hot['wait'], slots,
        rare['VALU'] + rare['SALU'] + rare['VMEM'], nbytes, res.get('VGPRs', '?'),
        res.get('ScratchSize', '?'), res.get('Occupancy', '?'),
        '%.2f-%.2f' % (LANES * 2.53 / valu, LANES * 3.13 / valu),
        '%.2f-%.2f' % (LANES * 2.53 / slots, LANES * 3.13 / slots)))


PROBES = {  # kernel: (instruction that must appear 64 times per iteration, extra, count)
    'probe_add': ('v_add_nc_u32', None, 0), 'probe_alignbit': ('v_alignbit_b32', None, 0),
    'probe_xor3': ('v_xor3_b32', None, 0), 'probe_add3': ('v_add3_u32', None, 0),
    'probe_bfi': ('v_bfi_b32', None, 0), 'probe_xad': ('v_xad_u32', None, 0),
    'probe_mix_salu': ('v_add_nc_u32', 's_xor_b32', 64),
    'probe_mix_delay': ('v_add_nc_u32', 's_delay_alu', 64),
}


def check_probes(clang, objdump, mcpu):
    """The issue-rate probes measure what they claim only if each loop
    iteration is exactly 64 of the probed instruction (plus the mix)."""
    text = ('#define get_local_id(d) __builtin_amdgcn_workitem_id_x()\n'
            '#define get_group_id(d) __builtin_amdgcn_workgroup_id_x()\n'
            '#define get_local_size(d) 64u\n' + open(os.path.join(ROOT, 'gpu/probe.cl')).read())
    asm, _ = compile_cl(clang, objdump, text, mcpu, [])
    funcs, ok, bad = parse(asm), True, []
    for name, (want, extra, n_extra) in PROBES.items():
        ins = funcs.get(name, [])
        loop = main_loop(ins) if ins else None
        ops = collections.Counter(op.split('_e32')[0].split('_e64')[0] for addr, op, size, tgt in ins
                                  if loop and loop[0] <= addr <= loop[1])
        valu = sum(n for op, n in ops.items() if op.startswith('v_'))
        if ops[want] != 64 or valu != 64 or (extra and ops[extra] < n_extra):
            bad.append('%s (%s x%d, VALU %d)' % (name, want, ops[want], valu))
    print('\nissue-rate probes: %s' % ('each loop is 64 of its instruction' if not bad
                                         else 'FAIL: ' + ', '.join(bad)))
    return not bad


def find_kernel(funcs, name):
    # Driver binaries may decorate kernel symbols; match by substring.
    found = [f for f in funcs if f == name] or \
        [f for f in funcs if name in f and 'sched' not in f]
    return found[0] if found else None


def main():
    args = sys.argv[1:]
    mcpu = args[args.index('--mcpu') + 1] if '--mcpu' in args else 'gfx1200'
    binary = args[args.index('--binary') + 1] if '--binary' in args else None
    clang, objdump = find_tools(need_clang=binary is None)
    if binary:
        r = subprocess.run([objdump, '-d', binary], capture_output=True, text=True)
        if r.returncode or '<fbm_' not in r.stdout:
            sys.exit('%s: could not disassemble fbm kernels (not an AMDGPU code object?)\n%s'
                     % (binary, r.stderr[-500:]))
        asm, res = r.stdout, notes_resources(binary)
        m = re.search(r'file format (\S+)', asm)
        print('driver-compiled binary %s (%s)' % (binary, m.group(1) if m else '?'))
    else:
        text = ''.join(open(os.path.join(ROOT, p)).read() for p in GPU_CL)
        asm, res = compile_cl(clang, objdump, text, mcpu, ['-DFBM_AUDIT', '-DFBM_WG=%d' % WG])
        print('%s, -mcpu=%s, work-group %d' % (os.path.basename(clang), mcpu, WG))
    funcs = parse(asm)
    gen = gen_counts()

    print('\nper hash = per loop iteration of one lane. GH/s = %d lanes x 2.53-3.13 GHz / '
          'instructions,\ncounting VALU only, or every instruction as one issue slot '
          '(pessimistic).\n' % LANES)
    print(HEADER)
    ok, got = True, {}
    for name, mode in (('fbm_nonce', 'nonce'), ('fbm_vr', 'vr')):
        fname = find_kernel(funcs, name)
        loop = main_loop(funcs[fname]) if fname else None
        if not loop:
            print('missing kernel or loop: %s (symbols: %s)' % (name, ', '.join(funcs)))
            ok = False
            continue
        hot, rare, ops, nbytes = region_stats(funcs[fname], *loop)
        got[mode] = hot
        g = gen[mode]['vector_instructions']
        r = res.get(fname, {})
        row(name, hot, rare, nbytes, r, g)
        if abs(hot['VALU'] - g) > TOLERANCE * g:
            print('  FAIL: %s loop has %d VALU, generator says %d (tolerance %d%%)'
                  % (name, hot['VALU'], g, 100 * TOLERANCE))
            ok = False
        if r.get('ScratchSize', '0') not in ('0', '?') or int(r.get('Occupancy', '16')) < 4:
            print('  FAIL: %s uses scratch or runs below 4 waves/SIMD' % name)
            ok = False
        if nbytes > ICACHE:
            print('  FAIL: %s loop is %d bytes, more than the %d-byte I$' % (name, nbytes, ICACHE))
            ok = False
        if mode == 'vr' and (hot['VMEM'] or not 0 < hot['SMEM'] <= 8):
            print('  FAIL: the vr loop must read its schedule with a few scalar loads '
                  '(SMEM %d, VMEM %d)' % (hot['SMEM'], hot['VMEM']))
            ok = False
        top = ', '.join('%s %d' % kv for kv in ops.most_common(9))
        print('|  | %s |' % top + ' |' * 14)
    if 'nonce' in got and 'vr' in got:
        n, v = got['nonce'], got['vr']
        print('\nvr vs nonce: %.3fx fewer VALU; %.3fx if every instruction takes an issue slot'
              % (n['VALU'] / v['VALU'], issue_slots(n) / issue_slots(v)))
    if binary:
        for fname, r in res.items():
            if 'fbm' in fname:
                print('%s: wave%s, %s VGPRs, scratch %s, ~%s waves/SIMD'
                      % (fname, r['wave'], r['VGPRs'], r['ScratchSize'], r['Occupancy']))

    if not binary:
        ok &= check_probes(clang, objdump, mcpu)

    base_dir = os.path.join(ROOT, 'bench/gpu-baselines/src')
    if not binary and os.path.isdir(base_dir):
        print('\nPrior art, cgminer 3.7.2 (one nonce per work-item; the whole kernel):\n')
        print(HEADER)
        base = []
        for b in BASELINES:
            p = os.path.join(base_dir, b)
            if not os.path.exists(p):
                continue
            basm, bres = compile_cl(clang, objdump, SHIM + open(p).read(), mcpu, [])
            ins = parse(basm)['search']
            hot, rare, ops, nbytes = region_stats(ins, ins[0][0], ins[-1][0])
            row(b.replace('.cl', ''), hot, rare, nbytes, bres.get('search', {}))
            base.append((b.replace('.cl', ''), hot))
        if base and 'vr' in got:
            v = got['vr']
            bv = min(base, key=lambda x: x[1]['VALU'])
            bs = min(base, key=lambda x: issue_slots(x[1]))
            print('\nvr vs the best of them: %.3fx fewer VALU (vs %s); %.3fx fewer instructions '
                  'if every one takes an issue slot (vs %s)'
                  % (bv[1]['VALU'] / v['VALU'], bv[0], issue_slots(bs[1]) / issue_slots(v), bs[0]))
    print('\nok' if ok else '\nFAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
