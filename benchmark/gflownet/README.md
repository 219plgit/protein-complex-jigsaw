# Amortised GFlowNet vs exact flow

Benchmark reported in the Bioinformatics manuscript (Section 3.6, Fig. 3, Supplementary S6).

```bash
pip install torch numpy matplotlib
python check_prop1.py            # seed-anchored histories == seed-caterpillar merger trees
python check_snare_variants.py   # typed SNARE (756 / 1,242) vs seating model (252 / 536)
python run_snare.py 3000         # SNARE, 1,242 trees, uniform and rugged rewards, 3 seeds
python run_large.py 2000 10000   # synthetic n=14, 3.6e8 trees, 3 seeds
python run_large_long.py 10000 0.3   # rugged reward, longer training with more exploration
python make_figure.py            # Fig. 3 (needs large_results.json)
```

`models.py` contains an independent explicit merger-tree enumerator used to cross-check
the released engine. The rugged reward is a synthetic stress test with no biological meaning.
