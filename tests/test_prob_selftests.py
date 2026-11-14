"""Run the built-in self-tests of pcjigsaw.prob (formerly sj_prob.py) under pytest,
so they cannot silently drift from the counters they check."""
import pytest

import pcjigsaw.prob as prob

SELF_TESTS = ["test_seed_recovery", "test_calibration", "test_support_identity",
              "test_merger_and_diagnostics", "test_second_order", "test_model_distribution",
              "test_counterfactual", "test_decision_value", "test_weighted_mergers"]


@pytest.mark.parametrize("name", SELF_TESTS)
def test_prob_self_test(name):
    getattr(prob, name)()
