#!/usr/bin/env python3
"""Generate unrolled SHA-256d mining kernels by partial evaluation.

Every intermediate value of SHA256(SHA256(header80)) is classified by how
often it changes:

    CONST  never: IV, round constants, padding words
    JOB    once per job: merkle-root tail, time, bits; also the midstate when
           every lane hashes the same version
    LANE   once per job but different in every SIMD lane: the midstate when
           lanes hold different rolled versions (BIP 320 / overt ASICBoost)
    NONCE  once per nonce but identical in every lane: the block-2 message
           schedule when lanes hold versions
    VAR    per lane and per nonce: the only work left in the SIMD loop

Operations on CONST inputs are folded at generation time. JOB, LANE and
NONCE operations are hoisted into scalar precompute functions. Only VAR
operations are emitted into the vector body. Sums are flat linear forms, so
constants combine across additions, and the vector additions that remain are
ordered by estimated readiness (Huffman on ready time) to shorten dependency
chains. The second hash stops as soon as H7 is known (e after round 60,
0-indexed), and dead code is removed.

Usage: gen_kernels.py OUTDIR       write one header per target/mode
       gen_kernels.py --ablation   print per-technique instruction counts
"""
import heapq
import json
import os
import sys
from functools import reduce

MASK = 0xFFFFFFFF

K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]
IV = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]


def rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & MASK


FUNCS = {
    'add': lambda a, b: (a + b) & MASK,
    'xor': lambda a, b: a ^ b,
    'and': lambda a, b: a & b,
    'bsig0': lambda x: rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22),
    'bsig1': lambda x: rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25),
    'ssig0': lambda x: rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3),
    'ssig1': lambda x: rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10),
    'ch': lambda e, f, g: (e & f) ^ (~e & g & MASK),
    'maj': lambda a, b, c: (a & b) ^ (a & c) ^ (b & c),
}
COMMUTATIVE = {'add', 'xor', 'and'}

CONST, JOB, LANE, NONCE, VAR = range(5)
KIND_NAME = ['const', 'job', 'lane', 'nonce', 'var']


def join(a, b):
    if a == b:
        return a
    lo, hi = min(a, b), max(a, b)
    if lo <= JOB:
        return hi
    return VAR  # LANE with NONCE, or anything with VAR


# Latency estimates (cycles), used only to order additions.
LATENCY = {
    'avx512': dict(add=1, xor=1, _and=1, bsig0=2, bsig1=2, ssig0=2, ssig1=2, ch=1, maj=1),
    'avx2': dict(add=1, xor=1, _and=1, bsig0=4, bsig1=4, ssig0=4, ssig1=4, ch=3, maj=3),
    'scalar': dict(add=1, xor=1, _and=1, bsig0=2, bsig1=2, ssig0=2, ssig1=2, ch=2, maj=2),
}
# Machine instructions per IR op, used only for the op-count report.
COST = {
    'avx512': dict(add=1, xor=1, _and=1, bsig0=4, bsig1=4, ssig0=4, ssig1=4, ch=1, maj=1),
    'avx2': dict(add=1, xor=1, _and=1, bsig0=11, bsig1=11, ssig0=9, ssig1=9, ch=3, maj=3),
    'scalar': dict(add=1, xor=1, _and=1, bsig0=5, bsig1=5, ssig0=5, ssig1=5, ch=3, maj=3),
}
# Targets with a single instruction for Ch/Maj (vpternlogd).
FUSED_TERNARY = {'avx512'}


def opkey(name):
    return '_and' if name == 'and' else name


class Atom:
    """A materialized value: an input, a constant, or one IR operation."""

    __slots__ = ('id', 'kind', 'op', 'args', 'value', 'name', 'ready', 'live')

    def __init__(self, gen, kind, op, args=(), value=None, name=None):
        self.id = len(gen.atoms)
        gen.atoms.append(self)
        self.kind, self.op, self.args, self.value, self.name = kind, op, tuple(args), value, name
        self.live = False
        self.ready = 0
        if kind == VAR and op not in ('input', 'const'):
            self.ready = max([a.ready for a in args if a.kind == VAR] + [0]) + gen.lat[opkey(op)]


class Lin:
    """A lazy sum of atoms plus a constant."""

    __slots__ = ('terms', 'c', 'atom')

    def __init__(self, terms=(), c=0):
        self.terms, self.c, self.atom = list(terms), c & MASK, None


def terms_of(x):
    if isinstance(x, int):
        return [], x & MASK
    if isinstance(x, Atom):
        return ([], x.value) if x.kind == CONST else ([x], 0)
    if x.atom is not None:
        return terms_of(x.atom)
    return list(x.terms), x.c


def add(*xs):
    terms, c = [], 0
    for x in xs:
        t, k = terms_of(x)
        terms += t
        c += k
    return Lin(terms, c)


class Gen:
    def __init__(self, target, mode, fold=True):
        self.target, self.mode, self.fold = target, mode, fold
        self.lat = LATENCY[target]
        self.atoms, self.cse, self.consts = [], {}, {}

    # -- construction ---------------------------------------------------
    def const(self, v):
        v &= MASK
        if v not in self.consts:
            # Without folding (ablation only), a constant is just another
            # vector operand: nothing is precomputed or simplified.
            self.consts[v] = Atom(self, CONST if self.fold else VAR, 'const', value=v)
        return self.consts[v]

    def input(self, name, kind):
        return Atom(self, kind, 'input', name=name)

    def op(self, name, *args):
        args = [self.mat(a) for a in args]
        if all(a.kind == CONST for a in args):
            return self.const(FUNCS[name](*[a.value for a in args]))
        if name in ('add', 'xor'):
            nz = [a for a in args if not (a.kind == CONST and a.value == 0)]
            if len(nz) == 1:
                return nz[0]
        if name == 'and' and any(a.kind == CONST and a.value == 0 for a in args):
            return self.const(0)
        if name in COMMUTATIVE:
            args = sorted(args, key=lambda a: a.id)
        key = (name, tuple(a.id for a in args))
        if key not in self.cse:
            kind = reduce(join, [a.kind for a in args])
            self.cse[key] = Atom(self, kind, name, args)
        return self.cse[key]

    def mat(self, x):
        """Materialize a Lin into one atom, folding everything uniform first."""
        if isinstance(x, int):
            return self.const(x)
        if isinstance(x, Atom):
            return x
        if x.atom is not None:
            return x.atom
        groups = {k: [] for k in (JOB, LANE, NONCE, VAR)}
        for t in x.terms:
            groups[t.kind].append(t)
        c = x.c
        # Fold JOB terms and the constant into the cheapest uniform group:
        # LANE (per job) before NONCE (per nonce) before JOB.
        carrier = (LANE if groups[LANE] else NONCE if groups[NONCE]
                   else JOB if groups[JOB] else None)
        if carrier in (LANE, NONCE):
            groups[carrier] += groups[JOB]
            groups[JOB] = []
        parts = []
        for k in (JOB, LANE, NONCE):
            if groups[k]:
                parts.append(self.chain(groups[k], c if k == carrier else 0))
        if carrier is None and (c or not groups[VAR]):
            parts.append(self.const(c))
        x.atom = self.tree(groups[VAR] + parts)
        return x.atom

    def chain(self, atoms, c):
        atoms = sorted(atoms, key=lambda a: a.id)
        acc = atoms[0]
        for a in atoms[1:]:
            acc = self.op('add', acc, a)
        return self.op('add', acc, self.const(c)) if c else acc

    def tree(self, atoms):
        heap = [(a.ready, a.id, a) for a in atoms]
        heapq.heapify(heap)
        while len(heap) > 1:
            x = heapq.heappop(heap)[2]
            y = heapq.heappop(heap)[2]
            s = self.op('add', x, y)
            heapq.heappush(heap, (s.ready, s.id, s))
        return heap[0][2]

    # -- SHA-256 on the IR ------------------------------------------------
    def ch(self, e, f, g):
        if self.target in FUSED_TERNARY:
            return self.op('ch', e, f, g)
        e, f, g = self.mat(e), self.mat(f), self.mat(g)
        return self.op('xor', g, self.op('and', e, self.op('xor', f, g)))

    def maj(self, a, b, c):
        if self.target in FUSED_TERNARY:
            return self.op('maj', a, b, c)
        a, b, c = self.mat(a), self.mat(b), self.mat(c)
        # b ^ ((a ^ b) & (b ^ c)): (b ^ c) is last round's (a ^ b), shared by CSE.
        return self.op('xor', b, self.op('and', self.op('xor', a, b), self.op('xor', b, c)))

    def expand(self, W, upto):
        while len(W) < upto:
            t = len(W)
            W.append(self.mat(add(self.op('ssig1', W[t - 2]), W[t - 7],
                                  self.op('ssig0', W[t - 15]), W[t - 16])))

    def rounds(self, state, W, t_from, t_to):
        a, b, c, d, e, f, g, h = state
        hk = add(h, K[t_from], W[t_from])
        for t in range(t_from, t_to):
            # g becomes the next round's h. Adding it into that round's
            # h + K + W *before* Ch makes Ch its last use, so vpternlogd can
            # overwrite g instead of needing a register copy.
            if t + 1 < t_to and self.mat(g).kind == VAR:
                hk_next = self.mat(add(g, K[t + 1], W[t + 1]))
            else:
                hk_next = add(g, K[t + 1], W[t + 1]) if t + 1 < t_to else None
            t1 = add(hk, self.op('bsig1', e), self.ch(e, f, g))
            if sum(1 for x in terms_of(t1)[0] if x.kind == VAR) >= 2:
                t1 = self.mat(t1)  # shared by e and a: compute once
            t2 = add(self.op('bsig0', a), self.maj(a, b, c))
            h, g, f, e, d, c, b, a = g, f, e, add(d, t1), c, b, a, add(t1, t2)
            hk = hk_next
        return [a, b, c, d, e, f, g, h]


def build(target, mode):
    g = Gen(target, mode)
    mid_kind = JOB if mode == 'nonce' else LANE
    w3_kind = VAR if mode == 'nonce' else NONCE
    mid = [g.input('mid%d' % i, mid_kind) for i in range(8)]
    tail = [g.input('w%d' % i, JOB) for i in range(3)]
    w3 = g.input('w3', w3_kind)

    # Hash 1, block 2: merkle tail, time, bits, nonce, then 80-byte padding.
    W = tail + [w3, 0x80000000] + [0] * 10 + [0x280]
    g.expand(W, 64)
    st = g.rounds(mid, W, 0, 64)
    digest = [g.mat(add(m, s)) for m, s in zip(mid, st)]

    # Hash 2: one block, 32-byte message then 32-byte padding. Only H7 is
    # needed: H7 = IV7 + (e after round 60), so rounds 61..63 are skipped.
    W2 = digest + [0x80000000] + [0] * 6 + [0x100]
    g.expand(W2, 61)
    st2 = g.rounds(list(IV), W2, 0, 61)
    h7 = g.mat(add(IV[7], st2[4]))
    return g, h7


# -- emission -------------------------------------------------------------
SCALAR_EXPR = {
    'add': '({0} + {1})', 'xor': '({0} ^ {1})', 'and': '({0} & {1})',
    'bsig0': 'S_BSIG0({0})', 'bsig1': 'S_BSIG1({0})',
    'ssig0': 'S_SSIG0({0})', 'ssig1': 'S_SSIG1({0})',
    'ch': 'S_CH({0}, {1}, {2})', 'maj': 'S_MAJ({0}, {1}, {2})',
}
VECTOR_EXPR = {
    'add': 'V_ADD({0}, {1})', 'xor': 'V_XOR({0}, {1})', 'and': 'V_AND({0}, {1})',
    'bsig0': 'V_BSIG0({0})', 'bsig1': 'V_BSIG1({0})',
    'ssig0': 'V_SSIG0({0})', 'ssig1': 'V_SSIG1({0})',
    'ch': 'V_CH({0}, {1}, {2})', 'maj': 'V_MAJ({0}, {1}, {2})',
}
INPUT_INDEX = dict({'mid%d' % i: i for i in range(8)}, w0=8, w1=9, w2=10)


def emit(g, root, prefix):
    # Liveness from the root.
    stack = [root]
    while stack:
        a = stack.pop()
        if a.live:
            continue
        a.live = True
        stack.extend(a.args)
    live = [a for a in g.atoms if a.live]

    # Atoms consumed by a later section are exported through a slot.
    slots = {JOB: {}, LANE: {}, NONCE: {}}
    for a in live:
        for x in a.args:
            if x.kind in slots and x.kind != a.kind and x.op != 'const' and x.id not in slots[x.kind]:
                slots[x.kind][x.id] = len(slots[x.kind])
    if root.kind in slots:
        raise SystemExit('root must be per-lane')
    arr = {JOB: 'J', LANE: 'L', NONCE: 'N'}
    local = {JOB: 'j', LANE: 'l', NONCE: 'n', VAR: 'v'}

    def ref(x, section):
        if x.op == 'const':
            lit = '0x%08xu' % x.value
            return 'V_SET1(%s)' % lit if section == VAR else lit
        if x.op == 'input' and x.kind == section:
            if x.name == 'w3':
                return 'w3'
            return 'in[%d]' % INPUT_INDEX[x.name]
        if x.kind == section:
            return '%s%d' % (local[section], x.id)
        s = '%s[%d]' % (arr[x.kind], slots[x.kind][x.id])
        if section == VAR and x.kind in (JOB, NONCE):
            return 'V_SET1(%s)' % s
        return s

    counts = {k: {} for k in (JOB, LANE, NONCE, VAR)}
    bodies = {k: [] for k in (JOB, LANE, NONCE, VAR)}
    for a in live:
        if a.op in ('input', 'const'):
            continue
        counts[a.kind][a.op] = counts[a.kind].get(a.op, 0) + 1
        table = VECTOR_EXPR if a.kind == VAR else SCALAR_EXPR
        ctype = 'V' if a.kind == VAR else 'uint32_t'
        expr = table[a.op].format(*[ref(x, a.kind) for x in a.args])
        bodies[a.kind].append('    const %s %s%d = %s;' % (ctype, local[a.kind], a.id, expr))
        if a.kind in slots and a.id in slots[a.kind]:
            bodies[a.kind].append('    %s[%d] = %s%d;' % (arr[a.kind], slots[a.kind][a.id], local[a.kind], a.id))
    # Exported inputs (e.g. the nonce word itself used by the vector body).
    for a in live:
        if a.op == 'input' and a.kind in slots and a.id in slots[a.kind]:
            bodies[a.kind].insert(0, '    %s[%d] = %s;' % (arr[a.kind], slots[a.kind][a.id], ref(a, a.kind)))

    def weighted(kind):
        return sum(COST[g.target][opkey(o)] * n for o, n in counts[kind].items())

    stats = {
        'target': g.target, 'mode': g.mode,
        'slots': {KIND_NAME[k]: len(v) for k, v in slots.items()},
        'ops': {KIND_NAME[k]: dict(sorted(v.items())) for k, v in counts.items()},
        'vector_instructions': weighted(VAR),
        'nonce_scalar_ops': sum(counts[NONCE].values()),
    }

    P = prefix
    out = []
    out.append('/* Generated by gen/gen_kernels.py. Do not edit. */')
    out.append('/* target=%s mode=%s */' % (g.target, g.mode))
    out.append('/* vector ops by kind: %s */' % json.dumps(stats['ops']['var'], sort_keys=True))
    out.append('/* ~%d vector instructions per body; %d per-nonce scalar ops */'
               % (stats['vector_instructions'], stats['nonce_scalar_ops']))
    out.append('#define %s_NJ %d' % (P.upper(), max(1, len(slots[JOB]))))
    out.append('#define %s_NL %d' % (P.upper(), max(1, len(slots[LANE]))))
    out.append('#define %s_NN %d' % (P.upper(), max(1, len(slots[NONCE]))))
    out.append('')
    out.append('/* in[0..7] = midstate, in[8..10] = block-2 words W0..W2 */')
    out.append('static inline void %s_job(const uint32_t *in, uint32_t *J)' % P)
    out.append('{')
    out.append('    (void)in; (void)J;')
    out += bodies[JOB]
    out.append('}')
    out.append('')
    out.append('static inline void %s_lane(const uint32_t *in, const uint32_t *J, uint32_t *L)' % P)
    out.append('{')
    out.append('    (void)in; (void)J; (void)L;')
    out += bodies[LANE]
    out.append('}')
    out.append('')
    out.append('static inline void %s_nonce(uint32_t w3, const uint32_t *J, uint32_t *N)' % P)
    out.append('{')
    out.append('    (void)w3; (void)J; (void)N;')
    out += bodies[NONCE]
    out.append('}')
    out.append('')
    out.append('/* Returns H7 of the final hash (big-endian word 7) for every lane. */')
    out.append('static inline V %s_body(const uint32_t *J, const V *L, const uint32_t *N, V w3)' % P)
    out.append('{')
    out.append('    (void)J; (void)L; (void)N; (void)w3;')
    out += bodies[VAR]
    out.append('    return %s;' % ref(root, VAR))
    out.append('}')
    return '\n'.join(out) + '\n', stats


def build_ablation(target, midstate, fold, early_exit, mode='nonce'):
    """The same computation with techniques switched off, for op counts."""
    g = Gen(target, mode, fold=fold)
    kind = (lambda k: k) if fold else (lambda k: VAR)
    const = (lambda v: v) if fold else g.const
    tail = [g.input('w%d' % i, kind(JOB)) for i in range(3)]
    w3 = g.input('w3', kind(VAR if mode == 'nonce' else NONCE))
    iv = [const(v) for v in IV]
    if midstate:
        mid = [g.input('mid%d' % i, kind(JOB if mode == 'nonce' else LANE)) for i in range(8)]
    else:  # hash the first 64 header bytes again for every nonce
        B = [g.input('b%d' % i, VAR) for i in range(16)]
        g.expand(B, 64)
        mid = [g.mat(add(v, s)) for v, s in zip(iv, g.rounds(iv, B, 0, 64))]
    W = tail + [w3, const(0x80000000)] + [const(0)] * 10 + [const(0x280)]
    g.expand(W, 64)
    digest = [g.mat(add(m, s)) for m, s in zip(mid, g.rounds(mid, W, 0, 64))]
    W2 = digest + [const(0x80000000)] + [const(0)] * 6 + [const(0x100)]
    if early_exit:
        g.expand(W2, 61)
        roots = [g.mat(add(iv[7], g.rounds(iv, W2, 0, 61)[4]))]
    else:
        g.expand(W2, 64)
        roots = [g.mat(add(v, s)) for v, s in zip(iv, g.rounds(iv, W2, 0, 64))]
    live, stack = set(), list(roots)
    while stack:
        a = stack.pop()
        if a.id not in live:
            live.add(a.id)
            stack.extend(a.args)
    ops = [a for a in g.atoms if a.id in live and a.op not in ('input', 'const')]
    vec = sum(COST[target][opkey(a.op)] for a in ops if a.kind == VAR)
    per_nonce = sum(COST['scalar'][opkey(a.op)] for a in ops if a.kind == NONCE)
    return vec, per_nonce


def ablation():
    """Deterministic per-technique instruction counts (no timing noise)."""
    steps = [
        ('naive: 3 compressions per nonce, nothing precomputed', False, False, False, 'nonce'),
        ('+ midstate (T1): 2 compressions per nonce', True, False, False, 'nonce'),
        ('+ constant folding / precompute (T2, T3)', True, True, False, 'nonce'),
        ('+ early exit after round 60 of hash 2 (T4)', True, True, True, 'nonce'),
        ('+ version rolling, 128 versions per nonce (T7)', True, True, True, 'vr'),
    ]
    for target, lanes in (('avx512', 16), ('avx2', 8)):
        label = {'avx512': 'AVX-512', 'avx2': 'AVX2'}[target]
        print('\n**%s (%d lanes)**, vector instructions per hash:\n' % (label, lanes))
        print('| step | per hash | vs previous | total reduction |\n|---|---:|---:|---:|')
        first = prev = None
        for name, midstate, fold, early, mode in steps:
            vec, per_nonce = build_ablation(target, midstate, fold, early, mode)
            per_hash = vec / lanes
            note = ''
            if per_nonce:
                note = ' (+%.1f scalar per hash for the shared schedule)' % (per_nonce / 128)
            first = first or per_hash
            print('| %s | %.1f%s | %s | %s |' % (
                name, per_hash, note,
                '%.1f%%' % (100 * (per_hash / prev - 1)) if prev else '-',
                '%.2fx' % (first / per_hash) if prev else '-'))
            prev = per_hash


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--ablation':
        return ablation()
    outdir = sys.argv[1] if len(sys.argv) > 1 else 'src/gen'
    os.makedirs(outdir, exist_ok=True)
    all_stats = []
    for target in ('scalar', 'avx2', 'avx512'):
        for mode in ('nonce', 'vr'):
            g, root = build(target, mode)
            prefix = 'g_%s_%s' % (target, mode)
            text, stats = emit(g, root, prefix)
            with open(os.path.join(outdir, prefix + '.h'), 'w') as f:
                f.write(text)
            all_stats.append(stats)
            print('%-7s %-6s vector_instr=%5d nonce_scalar_ops=%4d slots=%s' % (
                target, mode, stats['vector_instructions'], stats['nonce_scalar_ops'],
                stats['slots']))
    with open(os.path.join(outdir, 'stats.json'), 'w') as f:
        json.dump(all_stats, f, indent=1, sort_keys=True)


if __name__ == '__main__':
    main()
