# Experimental PCJ-HGTFlow research preview

This directory is **not part of the stable `pcjigsaw` API**. It contains a constraint-aware graph-attention policy trained on synthetic contact graphs by **exact-flow distillation**. It is **not a fully trained trajectory-balance GFlowNet** and it has **not been calibrated against experimentally measured assembly orders**.

Core code and checkpoints:
- `hgtflow/prototype.py`: exact teacher over connected binary merger trees and graph-attention candidate-split scorer;
- `hgtflow/prototype_results*.pt`: three frozen synthetic checkpoints;
- `extended_synthetic_benchmark.py`: deterministic seeded benchmarks against exact teachers, uniform and local-potential policies;
- `worked_synthetic_example.py`: modular six-protein contact deletion example;
- `analyse_7arc_hgt.py`: extract physical chain contacts from user-supplied `7ARC.cif` and exploratory neural sampling.

Install optionally from repository root:

```bash
python -m pip install -e '.[dev,hgtflow]'
python research/hgtflow/test_prototype.py
cd research
python extended_synthetic_benchmark.py --graphs-per-size 8 --output results/reproduced
python worked_synthetic_example.py --help
python analyse_7arc_hgt.py --cif /path/to/7ARC.cif --help
```

Data source: https://www.rcsb.org/structure/7ARC (Polytomella complex I peripheral arm, 16 protein chains). Obtain `7ARC.cif` yourself from https://files.rcsb.org/download/7ARC.cif. The repository includes derived contacts, not the CIF coordinates.

## Limits of interpretation

- Geometrical contacts are **candidate interfaces**, not temporal prerequisites.
- Neural samples use a limited shortlist of legal splits; their probabilities do not describe the full model space.
- The synthetic benchmark measures **local branching** total variation, not complete pathway-distribution accuracy.
- Model scores are synthetic preferences, **not** experimental binding energies or cellular assembly rates.
- No demonstrated generalization to new proteins or a calibrated biological 7ARC assembly order.

The base PCJ exact algorithms, the `pcjigsaw` namespace, and its compatibility aliases remain unchanged in this preview.
