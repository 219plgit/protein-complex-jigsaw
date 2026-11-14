"""Seed-anchored histories are the seed-caterpillar merger trees (Proposition 1).

The seed-anchored rule requires every addition to satisfy its prerequisites,
violate no exclusion and have an admitted contact with the current complex.
These tests check that rule against an independent explicit enumeration of
merger trees, and that legacy `requires` encodings are unchanged by it.
"""
import os
import sys
from itertools import permutations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sj_prob import TypedModel, seeded_F, merger_F            # noqa: E402
from scijigsaw.assembly import Assembly                      # noqa: E402

SNARE_CON = [("VAMP2", "SNAP25"), ("VAMP2", "Syntaxin-1A"), ("VAMP2", "Complexin"),
             ("SNAP25", "Syntaxin-1A"), ("SNAP25", "Syt-1"), ("Syntaxin-1A", "Complexin"),
             ("Syntaxin-1A", "Syt-1"), ("Syntaxin-1A", "Munc18-1"),
             ("VAMP2", "Synaptophysin"), ("VAMP2", "SNCA")]
SNARE_PRE = [("VAMP2", "Complexin"), ("Syntaxin-1A", "Complexin"),
             ("SNAP25", "Syt-1"), ("Syntaxin-1A", "Syt-1"), ("Syntaxin-1A", "Munc18-1")]
EIF3_CON = [("Prt1", "Tif34"), ("Tif34", "Tif35"), ("Prt1", "Tif32"), ("Tif32", "Nip1")]
EIF3_PRE = [("Tif34", "Tif35")]


def typed(contacts, prereqs):
    V = tuple(sorted({u for e in contacts for u in e}))
    return TypedModel(V=V, C=frozenset(frozenset(e) for e in contacts), P=tuple(prereqs))


# ---------------- independent explicit merger-tree enumeration ----------------
def explicit_trees(contacts, prereqs):
    units = sorted({u for e in contacts for u in e})
    nb = {u: set() for u in units}
    for a, b in contacts:
        nb[a].add(b)
        nb[b].add(a)
    pre = {u: {a for a, b in prereqs if b == u} for u in units}

    def connected(S):
        S = set(S)
        start = next(iter(S))
        seen, stack = {start}, [start]
        while stack:
            for v in nb[stack.pop()] & S - seen:
                seen.add(v)
                stack.append(v)
        return seen == S

    def ok_merge(A, B):
        C = A | B
        return (any(nb[a] & B for a in A) and all(pre[q] <= C for q in C))

    memo = {}

    def trees(S):
        S = frozenset(S)
        if len(S) == 1:
            return [S]
        if S in memo:
            return memo[S]
        out, low, rest = [], min(S), sorted(S - {min(S)})
        for r in range(len(rest)):
            for mask in range(1 << len(rest)):
                if bin(mask).count("1") != r:
                    continue
                A = frozenset({low} | {rest[i] for i in range(len(rest)) if mask >> i & 1})
                B = S - A
                if B and connected(A) and connected(B) and ok_merge(A, B):
                    out += [(S, ta, tb) for ta in trees(A) for tb in trees(B)]
        memo[S] = out
        return out
    return trees(units)


def caterpillars(trees, seed):
    def root(t):
        return t if isinstance(t, frozenset) else t[0]

    def cat(t):
        if isinstance(t, frozenset):
            return True
        _, a, b = t
        side, other = (a, b) if seed in root(a) else (b, a)
        return seed in root(side) and len(root(other)) == 1 and cat(side)
    return sum(cat(t) for t in trees)


def test_typed_eif3_seeded_equals_caterpillars():
    m = typed(EIF3_CON, EIF3_PRE)
    T = explicit_trees(EIF3_CON, EIF3_PRE)
    assert len(T) == merger_F(m)(frozenset(m.V)) == 14
    assert seeded_F(m, "Prt1")(frozenset()) == caterpillars(T, "Prt1") == 6
    # the earlier prerequisite-only rule lets a component join a complex it
    # does not touch (e.g. Nip1 before Tif32) and overcounts
    assert seeded_F(m, "Prt1", contact_connected=False)(frozenset()) == 12


def test_typed_snare_seeded_equals_caterpillars():
    m = typed(SNARE_CON, SNARE_PRE)
    T = explicit_trees(SNARE_CON, SNARE_PRE)
    assert len(T) == 1242
    assert seeded_F(m, "VAMP2")(frozenset()) == caterpillars(T, "VAMP2") == 756


def test_seating_model_snare():
    pre = SNARE_PRE + [("SNAP25", "Syntaxin-1A")]
    m = typed(SNARE_CON, pre)
    T = explicit_trees(SNARE_CON, pre)
    assert len(T) == 536
    assert seeded_F(m, "VAMP2")(frozenset()) == caterpillars(T, "VAMP2") == 252


def test_assembly_contact_option():
    VAMP2_REQ = {"SNAP25": {"VAMP2"}, "Syntaxin-1A": {"VAMP2", "SNAP25"},
                 "Munc18-1": {"Syntaxin-1A"}, "Complexin": {"VAMP2", "Syntaxin-1A"},
                 "Syt-1": {"SNAP25", "Syntaxin-1A"}, "Synaptophysin": {"VAMP2"},
                 "SNCA": {"VAMP2"}}
    legacy = Assembly(requires=VAMP2_REQ, excludes=[({"SNAP25"}, {"AP180", "CALM"})],
                      seed="VAMP2")
    assert legacy.n_orders_permitted() == legacy.n_orders_permitted(contact_connected=True) == 252
    eif3 = Assembly(requires={"Prt1": set(), "Tif34": {"Prt1"}, "Tif35": {"Tif34"},
                              "Tif32": {"Prt1"}, "Nip1": {"Tif32"}}, seed="Prt1")
    assert eif3.n_orders_permitted(contact_connected=True) == 6
    typed_eif3 = Assembly(contacts=EIF3_CON, prerequisites=EIF3_PRE, seed="Prt1")
    assert typed_eif3.n_orders_permitted(contact_connected=True) == 6
    typed_snare = Assembly(contacts=SNARE_CON, prerequisites=SNARE_PRE, seed="VAMP2")
    assert typed_snare.n_orders_permitted(contact_connected=True) == 756
