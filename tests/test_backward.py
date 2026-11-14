"""Backward (disassembly) policies on the seed-anchored state graph."""
import math

from pcjigsaw.prob import TypedModel, seeded_F
from pcjigsaw import backward as B

C = frozenset(frozenset(e) for e in [("Prt1", "Tif34"), ("Tif32", "Prt1"), ("Tif35", "Tif34"), ("Tif32", "Nip1")])
EIF3 = TypedModel(V=("Tif32", "Prt1", "Nip1", "Tif35", "Tif34"), C=C, P=(("Tif34", "Tif35"),))
IFACE = {frozenset({"Tif32", "Prt1"}): 195, frozenset({"Tif32", "Nip1"}): 206,
         frozenset({"Tif35", "Tif34"}): 84, frozenset({"Prt1", "Tif34"}): 73}

SN_CON = [("VAMP2", "SNAP25"), ("VAMP2", "Syntaxin-1A"), ("VAMP2", "Complexin"),
          ("SNAP25", "Syntaxin-1A"), ("SNAP25", "Syt-1"), ("Syntaxin-1A", "Complexin"),
          ("Syntaxin-1A", "Syt-1"), ("Syntaxin-1A", "Munc18-1"),
          ("VAMP2", "Synaptophysin"), ("VAMP2", "SNCA")]
SN_PRE = [("VAMP2", "Complexin"), ("Syntaxin-1A", "Complexin"), ("SNAP25", "Syt-1"),
          ("Syntaxin-1A", "Syt-1"), ("Syntaxin-1A", "Munc18-1"), ("SNAP25", "Syntaxin-1A")]
VAMP2 = TypedModel(V=tuple(sorted({u for e in SN_CON for u in e})),
                   C=frozenset(frozenset(e) for e in SN_CON), P=tuple(SN_PRE))


def policies(m, seed):
    out = [B.counting_backward(m, seed), B.uniform_backward(m, seed)]
    if m is EIF3:
        out += [B.interface_backward(m, seed, IFACE, b) for b in (0.5, 1, 2)]
    return out


def test_histories_are_the_seed_anchored_space():
    assert len(B.histories(EIF3, "Prt1")) == seeded_F(EIF3, "Prt1")(frozenset()) == 6
    assert len(B.histories(VAMP2, "VAMP2")) == seeded_F(VAMP2, "VAMP2")(frozenset()) == 252


def test_states_have_several_parents():
    st = B.reachable_states(VAMP2, "VAMP2")
    assert sum(len(B.parents(VAMP2, "VAMP2", S, st)) > 1 for S in st) > 0


def test_forward_policy_from_detailed_balance_reproduces_backward_distribution():
    for m, seed in ((EIF3, "Prt1"), (VAMP2, "VAMP2")):
        st = B.reachable_states(m, seed)
        full = max(st, key=len)
        for pb in policies(m, seed):
            d = B.pathway_distribution(m, seed, pb)
            assert abs(sum(d.values()) - 1) < 1e-12
            pf, F = B.forward_from_backward(m, seed, pb)
            for S in st:
                if S != full:
                    assert abs(sum(pf(S).values()) - 1) < 1e-12
            for h, p in d.items():
                assert abs(B.history_probability_forward(h, pf) - p) < 1e-12


def test_counting_backward_is_the_unit_reward_flow():
    for m, seed, n in ((EIF3, "Prt1", 6), (VAMP2, "VAMP2", 252)):
        d = B.pathway_distribution(m, seed, B.counting_backward(m, seed))
        assert all(abs(p - 1 / n) < 1e-12 for p in d.values())


def test_uniform_disassembly_is_not_uniform_assembly():
    d = B.pathway_distribution(EIF3, "Prt1", B.uniform_backward(EIF3, "Prt1"))
    assert abs(B.effective_number(d) - 6) > 0.1


def test_admissibility_is_backward_policy_invariant():
    """Every admissible history keeps positive probability; nothing else gets any."""
    H = set(B.histories(EIF3, "Prt1"))
    for pb in policies(EIF3, "Prt1"):
        d = B.pathway_distribution(EIF3, "Prt1", pb)
        assert set(d) == H and all(p > 0 for p in d.values())


def test_interface_disassembly_values_reported_in_SI():
    d = B.pathway_distribution(EIF3, "Prt1", B.interface_backward(EIF3, "Prt1", IFACE, 1.0))
    assert abs(B.prefix_support(d, {"Tif34"}, "Prt1") - 0.132) < 5e-4
    assert abs(B.effective_number(d) - 3.452) < 5e-4
    d0 = B.pathway_distribution(EIF3, "Prt1", B.interface_backward(EIF3, "Prt1", IFACE, 0.0))
    du = B.pathway_distribution(EIF3, "Prt1", B.uniform_backward(EIF3, "Prt1"))
    assert all(abs(d0[h] - du[h]) < 1e-12 for h in du)              # beta = 0 is uniform


# ---------------- beta-sensitivity and backward-entropy diagnostics ----------------
EIF3_NO_PRE = TypedModel(V=EIF3.V, C=C, P=())
BETAS = (0, 0.5, 1, 2, 4)


def test_interface_conclusion_is_evidence_sensitive():
    s = B.beta_sensitivity(EIF3, "Prt1", lambda b: B.interface_backward(EIF3, "Prt1", IFACE, b), BETAS, {"Tif34"})
    assert abs(s["delta"] - 0.5) < 1e-3                       # reported in Supplementary Section S11


def test_count_score_signal_comes_only_from_the_redundant_prerequisite():
    """Same six histories without Tif34<Tif35; the count score then carries no information."""
    assert len(B.histories(EIF3_NO_PRE, "Prt1")) == 6
    for b in BETAS:
        pb = B.relation_count_backward(EIF3_NO_PRE, "Prt1", b)
        full = max(B.reachable_states(EIF3_NO_PRE, "Prt1"), key=len)
        assert abs(B.backward_entropy(pb, full) - math.log(2)) < 1e-12
    s = B.beta_sensitivity(EIF3_NO_PRE, "Prt1", lambda b: B.relation_count_backward(EIF3_NO_PRE, "Prt1", b),
                           BETAS, {"Tif34"})
    assert s["delta"] < 1e-12


def test_scores_disagree_on_the_first_dissociation():
    full = max(B.reachable_states(EIF3, "Prt1"), key=len)
    assert B.interface_backward(EIF3, "Prt1", IFACE, 1)(full)["Tif35"] > 0.5
    assert B.relation_count_backward(EIF3, "Prt1", 1)(full)["Nip1"] > 0.5


# ---------------- usefulness of evidence (linear response at beta = 0) ----------------
def _fd_slope(model, score, block, scale, e=1e-5):
    std = lambda p, S: score(p, S) / scale
    sp = lambda b: B.prefix_support(B.pathway_distribution(model, "Prt1", B.score_backward(model, "Prt1", std, b)),
                                    block, "Prt1")
    return (sp(e) - sp(0)) / e


def test_usefulness_formula_matches_finite_difference():
    for model, score in ((EIF3, B.interface_score(EIF3, "Prt1", IFACE)), (EIF3, B.relation_count_score(EIF3, "Prt1"))):
        for block in ({"Tif34"}, {"Tif34", "Tif35"}, {"Tif32"}):
            u = B.evidence_usefulness(model, "Prt1", score, block)
            assert abs(u["U"] - _fd_slope(model, score, block, u["scale"])) < 1e-4


def test_usefulness_values_reported_in_SI():
    ui = B.evidence_usefulness(EIF3, "Prt1", B.interface_score(EIF3, "Prt1", IFACE), {"Tif34"})
    uc = B.evidence_usefulness(EIF3, "Prt1", B.relation_count_score(EIF3, "Prt1"), {"Tif34"})
    u0 = B.evidence_usefulness(EIF3_NO_PRE, "Prt1", B.relation_count_score(EIF3_NO_PRE, "Prt1"), {"Tif34"})
    assert abs(ui["U"] + 0.749) < 1e-3 and ui["S_inf"] < 1e-6
    assert abs(uc["U"] - 0.484) < 1e-3 and uc["S_inf"] > 1 - 1e-6
    assert u0["U"] == 0.0 and abs(u0["S_inf"] - 0.5) < 1e-12
