from __future__ import annotations

import math
import random

from pcjigsaw.flow import (ExactMergerFlow, ExactSeedFlow, compare_model_flows,
    plackett_luce_step_weight, interface_potential)


class AssemblyLike:
    def __init__(self, *, requires, contacts=(), excludes=(), seed=None, units=None):
        self.requires = {k: set(v) for k, v in requires.items()}
        self.seed = seed
        self.excludes = [(set(a), set(b)) for a, b in excludes]
        self.units = list(units if units is not None else requires)
        implied = {frozenset((u, r)) for u, rs in self.requires.items() for r in rs}
        self._contacts = implied | {frozenset(x) for x in contacts}


def test_seed_uniform_flow_is_exact_history_count():
    # A and B may arrive in either order, then C requires both: 2 histories.
    m = AssemblyLike(requires={"A": set(), "B": set(), "C": {"A", "B"}})
    f = ExactSeedFlow(m)
    assert f.flow() == 2
    rows = f.transitions()
    assert {r.component for r in rows} == {"A", "B"}
    assert all(math.isclose(r.probability, 0.5) for r in rows)


def test_seed_weight_redistributes_but_does_not_restore_forbidden_steps():
    m = AssemblyLike(requires={"A": set(), "B": set(), "C": {"A"}})
    f = ExactSeedFlow(m, step_weight=lambda S, u: 4.0 if u == "A" else 1.0)
    first = {r.component: r.probability for r in f.transitions()}
    assert first["A"] > first["B"]
    # C is inadmissible initially despite any reward design.
    assert "C" not in first


def test_merger_uniform_flow_counts_binary_trees():
    # Complete contact graph on 3 nodes has 3 rooted unordered binary trees.
    V = ("A", "B", "C")
    contacts = [("A", "B"), ("A", "C"), ("B", "C")]
    m = AssemblyLike(requires={v: set() for v in V}, contacts=contacts)
    f = ExactMergerFlow(m)
    assert f.flow() == 3
    assert len(f.splits()) == 3
    assert all(math.isclose(x.probability, 1/3) for x in f.splits())


def test_merger_support_reduces_to_exact_count_ratio():
    V = ("A", "B", "C")
    contacts = [("A", "B"), ("A", "C"), ("B", "C")]
    m = AssemblyLike(requires={v: set() for v in V}, contacts=contacts)
    f = ExactMergerFlow(m)
    r = f.support({"A", "B"})
    assert r["flow_containing"] == 1
    assert r["flow_total"] == 3
    assert math.isclose(r["support"], 1/3)


def test_merger_sampling_is_reward_proportional():
    V = ("A", "B", "C")
    contacts = [("A", "B"), ("A", "C"), ("B", "C")]
    m = AssemblyLike(requires={v: set() for v in V}, contacts=contacts)
    def w(A, B, S):
        # favour a root split where A-B is formed as the size-2 child
        pair = A if len(A) == 2 else B if len(B) == 2 else frozenset()
        return 5.0 if pair == frozenset({"A","B"}) else 1.0
    f = ExactMergerFlow(m, merge_weight=w)
    probs = {frozenset(row.left if len(row.left)==2 else row.right): row.probability
             for row in f.splits()}
    assert probs[frozenset({"A","B"})] > probs[frozenset({"A","C"})]


def test_edge_additive_degeneracy_is_explicit():
    V = ("A", "B", "C")
    contacts = [("A", "B"), ("A", "C"), ("B", "C")]
    m = AssemblyLike(requires={v: set() for v in V}, contacts=contacts)
    d = ExactMergerFlow(m).edge_additive_degeneracy()
    assert d["non_identifying"] is True
    assert d["constant_terminal_score"] == 3


def test_model_alternatives_are_kept_separate():
    m1 = AssemblyLike(requires={"A": set(), "B": {"A"}})
    m2 = AssemblyLike(requires={"A": set(), "B": set()})
    r = compare_model_flows({"ordered": m1, "free": m2}, "seed")
    assert [x.label for x in r] == ["ordered", "free"]
    assert [x.summary.root_flow for x in r] == [1, 2]


def test_vamp2_uniform_seed_flow_recovers_published_252():
    m = AssemblyLike(
        requires={
            "SNAP25": {"VAMP2"},
            "Syntaxin-1A": {"VAMP2", "SNAP25"},
            "Munc18-1": {"Syntaxin-1A"},
            "Complexin": {"VAMP2", "Syntaxin-1A"},
            "Syt-1": {"SNAP25", "Syntaxin-1A"},
            "Synaptophysin": {"VAMP2"},
            "SNCA": {"VAMP2"},
        },
        seed="VAMP2",
    )
    assert ExactSeedFlow(m).flow() == 252


def test_seeded_flow_requires_contact_with_current_complex():
    m = AssemblyLike(
        requires={"A": set(), "B": {"A"}},
        contacts=[("seed", "A")], seed="seed")
    # Here B<-A is a temporal prerequisite only, not a declared contact.
    m._contacts = {frozenset(("seed", "A"))}
    f = ExactSeedFlow(m)
    assert f.admissible_additions(frozenset()) == ("A",)
    # B satisfies its prerequisite after A, but is not physically connected to
    # the assembled complex by a declared contact and therefore cannot be added.
    assert f.admissible_additions(frozenset({"A"})) == ()


def test_plackett_luce_is_invariant_to_global_rescaling():
    units=("A","B","C")
    w1=plackett_luce_step_weight({"A":1,"B":2,"C":4},units)
    w2=plackett_luce_step_weight({"A":10,"B":20,"C":40},units)
    for state,u in [(frozenset(),"B"),(frozenset({"B"}),"A")]:
        assert math.isclose(w1(state,u),w2(state,u),rel_tol=1e-12)


def test_interface_potential_is_constant_on_a_path_graph():
    V=("A","B","C","D")
    contacts=[("A","B"),("B","C"),("C","D")]
    m=AssemblyLike(requires={v:set() for v in V}, contacts=contacts)
    rec={frozenset(e):1.0 for e in contacts}
    residues={frozenset(e):10.0 for e in contacts}
    f0=ExactMergerFlow(m, merge_weight=interface_potential(rec,residues,lam=0.0))
    f2=ExactMergerFlow(m, merge_weight=interface_potential(rec,residues,lam=2.0))
    # Every complete tree contains each edge at exactly one one-contact merge, so
    # lambda changes the common scale but not relative tree probabilities.
    assert len(f0.splits()) == len(f2.splits())
    p0=sorted(x.probability for x in f0.splits())
    p2=sorted(x.probability for x in f2.splits())
    assert all(math.isclose(a,b,rel_tol=1e-12) for a,b in zip(p0,p2))


def test_typed_encodings_do_not_add_the_seed():
    """Typed encodings list the seed among the units; it must not be added."""
    from scijigsaw.assembly import Assembly
    from pcjigsaw.flow import ExactSeedFlow, ExactMergerFlow
    eif3 = Assembly(contacts=[("Prt1", "Tif34"), ("Tif34", "Tif35"), ("Prt1", "Tif32"),
                              ("Tif32", "Nip1")], prerequisites=[("Tif34", "Tif35")], seed="Prt1")
    assert ExactSeedFlow(eif3).flow() == 6
    assert ExactMergerFlow(eif3).flow() == 14
    assert eif3.n_orders_permitted(contact_connected=True) == 6
