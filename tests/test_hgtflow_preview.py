"""Optional checks for the research preview. No stable API or biological claims."""
import sys
from pathlib import Path
import pytest

pytest.importorskip('torch')
pytest.importorskip('numpy')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research' / 'hgtflow'))
from prototype import ExactMerger, HierarchicalPolicy, modular_graph
import numpy as np
import torch


def test_connected_four_component_binary_mergers():
    adj = np.ones((4, 4), dtype=np.float32) - np.eye(4, dtype=np.float32)
    teacher = ExactMerger(adj, np.zeros((4, 4), dtype=np.float32))
    assert round(np.exp(teacher.logZ(0b1111))) == 15
    for mask, splits, probabilities in teacher.states_and_teacher():
        assert np.isclose(probabilities.sum(), 1, atol=1e-7)
        assert all(a & b == 0 and (a | b) == mask for a, b in splits)


def test_graph_attention_policy_outputs_finite_scores():
    import random
    adj, weights, _ = modular_graph(7, random.Random(2))
    mask, splits, _ = ExactMerger(adj, weights).states_and_teacher()[0]
    with torch.no_grad():
        scores = HierarchicalPolicy()(torch.tensor(adj), torch.tensor(weights), mask, splits)
    assert scores.numel() == len(splits)
    assert torch.isfinite(scores).all()
