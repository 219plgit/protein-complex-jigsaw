"""Forest (bottom-up, multi-parent) GFlowNet construction contributed for v12.

Pins the two properties stated in Supplementary Section S11.5: the backward
policy redistributes each tree's flow over that tree's own trajectories, so tree
probabilities, and hence subcomplex supports, do not depend on beta; and the
exact engine agrees with the research counter.
"""
import math
import os
import sys
from collections import defaultdict

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "benchmark", "gflownet", "forest"))
torch = pytest.importorskip("torch")

from model import eif3_model, snare_model            # noqa: E402
from exact import ExactMergerFlow                     # noqa: E402
from biobackward import ForestSampler, tree_key       # noqa: E402
from pcjigsaw.prob import TypedModel, merger_F        # noqa: E402


def research(m):
    tm = TypedModel(V=m.units, C=frozenset(frozenset(e) for e in m.contacts), P=tuple(m.prerequisites))
    return merger_F(tm)(frozenset(m.units))


def test_exact_engine_agrees_with_research_counter():
    for m, n in ((eif3_model(), 14), (snare_model(), 1242)):
        assert ExactMergerFlow(m).flow() == research(m) == n


def _trajectories(s, forest, path):
    if len(forest) == 1:
        yield forest[0], path
        return
    cand, _ = s.forward_logprobs(forest)
    for c in cand:
        yield from _trajectories(s, c[5], path + [(forest, c[5])])


@pytest.mark.parametrize("beta", [0.0, 1.0, 4.0])
def test_backward_policy_sums_to_one_over_each_trees_trajectories(beta):
    s = ForestSampler(eif3_model(), log_phi=lambda A, B: 0.0, fixed_beta=beta)
    per_tree = defaultdict(float)
    for tree, path in _trajectories(s, s.initial(), []):
        p = 1.0
        for parent, child in path:
            bc, lb = s.backward_logprobs(child)
            k = next(i for i, x in enumerate(bc) if x[3] == parent)
            p *= math.exp(float(lb[k]))
        per_tree[tree_key(tree)] += p
    assert len(per_tree) == 14
    assert all(abs(v - 1.0) < 1e-5 for v in per_tree.values())


def test_states_have_several_parents_and_beta_is_fixed_by_default():
    s = ForestSampler(eif3_model(), log_phi=lambda A, B: 0.0)
    assert s.fixed_beta == 1.0 and not s.policy.raw_beta.requires_grad
    multi = False
    for c1 in s.forward_logprobs(s.initial())[0]:
        for c2 in s.forward_logprobs(c1[5])[0]:
            bc, lp = s.backward_logprobs(c2[5])
            if len(bc) >= 2:
                multi = True
                assert abs(float(lp.exp().sum()) - 1) < 1e-6
    assert multi
