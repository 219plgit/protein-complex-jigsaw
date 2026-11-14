"""The teaching pack prints only numbers that agree with the research code.

`education/pcj_teach.py` has its own small, readable enumerator so students can
follow it. These tests pin its answers to the paper's values and to the
research counters in pcjigsaw.prob, so the worksheets can never drift from the paper.
"""
import os
import sys
from fractions import Fraction

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from pcjigsaw.education import TeachingModel, compute, build_pack, model_path   # noqa: E402
from pcjigsaw.education import compute_vamp2, buildable, board, build_vamp2_pack   # noqa: E402
from pcjigsaw.education.teach import DATA                                         # noqa: E402
from pcjigsaw.prob import TypedModel, seeded_F, merger_F                          # noqa: E402

MODELS = os.path.join(DATA, "models")


def research_counts(m: TeachingModel):
    tm = TypedModel(V=tuple(sorted(m.components)),
                    C=frozenset(frozenset(e) for e in m.contacts), P=())
    return seeded_F(tm, m.seed)(frozenset()), merger_F(tm)(frozenset(tm.V))


def test_eif3_matches_paper():
    m = TeachingModel.from_dir(os.path.join(MODELS, "eif3"))
    R = compute(m)
    assert R["seeded"]["count"] == 6
    assert R["merger"]["count"] == 14
    assert R["merger"]["query_support"] == [5, 14]
    assert R["seeded"]["query_support"] == [0, 6]
    assert sorted(R["merger"]["root_splits"].values()) == [2, 2, 5, 5]
    assert all(d["seeded"] == 0 and d["merger"] == 0 for d in R["perturbations"])


def test_agrees_with_research_counters():
    for name in ("eif3", "chain4", "cycle4"):
        m = TeachingModel.from_dir(os.path.join(MODELS, name))
        R = compute(m)
        assert (R["seeded"]["count"], R["merger"]["count"]) == research_counts(m), name


def test_proposition_1_and_flow_uniformity():
    for name in ("eif3", "chain4", "cycle4"):
        m = TeachingModel.from_dir(os.path.join(MODELS, name))
        R = compute(m)
        assert R["merger"]["seed_caterpillars"] == R["seeded"]["count"]
        n = R["seeded"]["count"]
        flows = [Fraction(d["flow"]) for d in R["seeded"]["naive_vs_flow"]]
        naive = [Fraction(d["naive"]) for d in R["seeded"]["naive_vs_flow"]]
        assert all(f == Fraction(1, n) for f in flows)
        assert sum(naive) == 1


def test_chain_and_ring_crossings():
    Rc = compute(TeachingModel.from_dir(os.path.join(MODELS, "chain4")))
    Rr = compute(TeachingModel.from_dir(os.path.join(MODELS, "cycle4")))
    assert Rc["merger"]["count"] == 5 and Rc["merger"]["crossing_count_patterns"] == {"[1, 1, 1]": 5}
    assert Rr["merger"]["count"] == 10 and Rr["merger"]["crossing_count_patterns"] == {"[1, 1, 2]": 10}
    assert len(Rr["merger"]["multi_contact_sets"]) == 6


def test_pack_builds(tmp_path):
    build_pack(os.path.join(MODELS, "eif3"), str(tmp_path), kits=True)
    for f in ("outreach_card.pdf", "worksheet.pdf", "answers.pdf", "answers.json",
              "kit_student.pdf", "kit_teacher.pdf"):
        assert (tmp_path / f).stat().st_size > 1000, f


# ---------------- VAMP2 activity (the v21 educational implementation) ----------------
build_vamp2 = build_vamp2_pack


def test_vamp2_answer_key_matches_research_model():
    R = compute_vamp2()
    assert R["orders_admissible"] == R["orders_admissible_assembly"] == R["orders_admissible_typed"] == 252
    assert R["orders_total"] == 5040
    assert R["merger_trees_typed"] == 536
    assert R["revision_1KIL"]["before"] == 336 and R["revision_1KIL"]["after"] == 252
    assert R["bridges"] == {"Syt-1": ["SNAP25", "Syntaxin-1A"], "Complexin": ["Syntaxin-1A", "VAMP2"]}
    assert R["syt1_complexin_contact"] is False
    assert sorted(list(R["alternatives"].values())[0]) == ["AP180", "CALM", "SNAP25"]
    q = {tuple(x["state"]): x for x in R["queries"]}
    assert q[("VAMP2", "SNAP25", "Syntaxin-1A")]["histories_through_state"] == 120   # stated in SI S10.1
    assert len(R["core_orders"]) == 1


def test_vamp2_semantic_invariance():
    """Exclusion becomes alternative occupancy; prerequisites become wait cues."""
    _, inter = board()
    assert buildable(["VAMP2", "AP180"], inter)[0]
    assert not buildable(["VAMP2", "SNAP25", "AP180"], inter)[0]
    assert not buildable(["VAMP2", "Complexin"], inter)[0]
    assert buildable(["VAMP2", "SNAP25", "Syntaxin-1A", "Syt-1", "Complexin"], inter)[0]


def test_vamp2_reward_redistributes_never_removes():
    R = compute_vamp2()["reward"]
    assert R["admissible_under_all_checked_priors"]
    assert R["rescaling_invariant"]
    assert R["reported_gtex"]["N_eff_v10"] == "57.8 ± 18.4"          # quoted from the main text, not recomputed


def test_vamp2_pack_builds(tmp_path):
    build_vamp2(str(tmp_path), kits=True)
    for f in ("vamp2_steps.pdf", "vamp2_activity.pdf", "vamp2_answers.pdf", "advanced_worksheet.pdf",
              "advanced_answers.pdf", "vamp2_answers.json", "kit_student.pdf", "kit_teacher.pdf"):
        assert (tmp_path / f).stat().st_size > 1000, f


# ---------------- one executable model: packaged data == repository data ----------------
import filecmp            # noqa: E402
import subprocess         # noqa: E402


def test_packaged_board_is_the_research_board():
    for f in ("proteins.csv", "interactions.csv"):
        assert filecmp.cmp(os.path.join(DATA, "vamp2_board", f),
                           os.path.join(ROOT, "examples", "vamp2", f), shallow=False), f


def test_packaged_benchmark_results_are_the_stored_results():
    for f in ("snare_results.json", "large_results.json", "large_long_10000_0.3.json"):
        assert filecmp.cmp(os.path.join(DATA, "gflownet", f),
                           os.path.join(ROOT, "benchmark", "gflownet", "results", f), shallow=False), f


def test_teaching_eif3_is_the_research_eif3():
    from test_seed_contact import EIF3_CON
    m = TeachingModel.from_dir(model_path("eif3"))
    assert {frozenset(e) for e in m.contacts} == {frozenset(e) for e in EIF3_CON}


def test_sj_prob_is_an_alias_of_the_packaged_module():
    import sj_prob
    import pcjigsaw.prob
    assert sj_prob is pcjigsaw.prob


def test_cli_check_reproduces_the_paper():
    r = subprocess.run([sys.executable, "-m", "pcjigsaw.education", "check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "FAIL" not in r.stdout
