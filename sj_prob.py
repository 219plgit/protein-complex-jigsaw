"""Compatibility alias: the research module now lives in pcjigsaw.prob.

`import sj_prob` returns the very same module object as `import pcjigsaw.prob`,
so every existing script, notebook and test (including private names) keeps
working unchanged. Running `python sj_prob.py` runs the module's self-tests.
"""
import os
import sys

try:
    import pcjigsaw.prob as _prob
except ImportError:                      # running from a checkout without `pip install -e .`
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
    import pcjigsaw.prob as _prob

if __name__ == "__main__":
    for _name in ("test_seed_recovery", "test_calibration", "test_support_identity",
                  "test_merger_and_diagnostics", "test_second_order", "test_model_distribution",
                  "test_counterfactual", "test_decision_value", "test_weighted_mergers"):
        getattr(_prob, _name)()
    print("\nall self-tests passed")
else:
    sys.modules[__name__] = _prob
