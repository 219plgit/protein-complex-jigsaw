"""Typed encodings used in the Bioinformatics manuscript, plus an independent
explicit merger-tree enumerator used to check the released engine."""
from itertools import permutations

# ---- neuronal SNARE / VAMP2 fusion state (typed; 1,242 merger trees) ----
SNARE_CON = [("VAMP2", "SNAP25"), ("VAMP2", "Syntaxin-1A"), ("VAMP2", "Complexin"),
             ("SNAP25", "Syntaxin-1A"), ("SNAP25", "Syt-1"), ("Syntaxin-1A", "Complexin"),
             ("Syntaxin-1A", "Syt-1"), ("Syntaxin-1A", "Munc18-1"),
             ("VAMP2", "Synaptophysin"), ("VAMP2", "SNCA")]
SNARE_PRE = [("VAMP2", "Complexin"), ("Syntaxin-1A", "Complexin"),
             ("SNAP25", "Syt-1"), ("Syntaxin-1A", "Syt-1"),
             ("Syntaxin-1A", "Munc18-1")]

# ---- legacy VAMP2 seating model (every partner is a prerequisite; 252 orders) ----
VAMP2_REQ = {"SNAP25": {"VAMP2"}, "Syntaxin-1A": {"VAMP2", "SNAP25"},
             "Munc18-1": {"Syntaxin-1A"}, "Complexin": {"VAMP2", "Syntaxin-1A"},
             "Syt-1": {"SNAP25", "Syntaxin-1A"}, "Synaptophysin": {"VAMP2"},
             "SNCA": {"VAMP2"}}

# ---- yeast eIF3 core (typed; 14 merger trees) ----
EIF3_CON = [("Prt1", "Tif34"), ("Tif34", "Tif35"), ("Prt1", "Tif32"), ("Tif32", "Nip1")]
EIF3_PRE = [("Tif34", "Tif35")]


class Model:
    def __init__(self, contacts, prereqs, units=None):
        self.units = sorted(units or {u for e in contacts for u in e} |
                            {u for e in prereqs for u in e})
        self.idx = {u: i for i, u in enumerate(self.units)}
        n = len(self.units)
        self.n = n
        self.con = [0] * n
        self.pre = [0] * n            # pre[v] = mask of units that must precede v
        for a, b in contacts:
            i, j = self.idx[a], self.idx[b]
            self.con[i] |= 1 << j
            self.con[j] |= 1 << i
        for a, b in prereqs:
            self.pre[self.idx[b]] |= 1 << self.idx[a]
        self.full = (1 << n) - 1

    def mask(self, names):
        return sum(1 << self.idx[u] for u in names)

    def connected(self, S):
        if S == 0:
            return False
        f = (S & -S).bit_length() - 1
        seen, stack = 1 << f, [f]
        while stack:
            i = stack.pop()
            nb = self.con[i] & S & ~seen
            while nb:
                j = (nb & -nb).bit_length() - 1
                seen |= 1 << j
                stack.append(j)
                nb &= nb - 1
        return seen == S

    def closed(self, S):
        m = S
        while m:
            i = (m & -m).bit_length() - 1
            if self.pre[i] & ~S:
                return False
            m &= m - 1
        return True

    def merge_ok(self, A, B):
        linked = any(self.con[i] & B for i in range(self.n) if A >> i & 1)
        return linked and self.closed(A | B)

    def splits(self, S):
        """Canonical unordered splits {A,B} of S (A holds the lowest bit)."""
        low = S & -S
        sub = (S - 1) & S
        while sub:
            if sub & low:
                A, B = sub, S ^ sub
                if B and self.connected(A) and self.connected(B) and self.merge_ok(A, B):
                    yield A, B
            sub = (sub - 1) & S

    # explicit enumeration: a tree is a nested tuple of masks
    def trees(self, S=None, memo=None):
        S = self.full if S is None else S
        memo = {} if memo is None else memo
        if S & (S - 1) == 0:
            return [S]
        if S in memo:
            return memo[S]
        out = []
        for A, B in self.splits(S):
            for ta in self.trees(A, memo):
                for tb in self.trees(B, memo):
                    out.append((S, ta, tb))
        memo[S] = out
        return out


def nodes(t):
    """All subcomplex masks appearing in tree t (internal nodes and leaves)."""
    if isinstance(t, int):
        return [t]
    return [t[0]] + nodes(t[1]) + nodes(t[2])


def is_seed_caterpillar(t, seed_bit):
    """Every merge joins the seed-containing branch to a single component."""
    if isinstance(t, int):
        return True
    _, a, b = t
    ma = a if isinstance(a, int) else a[0]
    mb = b if isinstance(b, int) else b[0]
    if ma & seed_bit:
        seedside, other, om = a, b, mb
    elif mb & seed_bit:
        seedside, other, om = b, a, ma
    else:
        return False
    return (om & (om - 1)) == 0 and is_seed_caterpillar(seedside, seed_bit)


def seeded_orders(m, seed, require_contact=True):
    """Seed-anchored histories: add one component at a time; prerequisites
    present and (optionally) a contact with the current complex."""
    others = [u for u in m.units if u != seed]
    count = 0
    for perm in permutations(others):
        S = m.mask([seed])
        ok = True
        for u in perm:
            i = m.idx[u]
            if m.pre[i] & ~S or (require_contact and not (m.con[i] & S)):
                ok = False
                break
            S |= 1 << i
        count += ok
    return count
