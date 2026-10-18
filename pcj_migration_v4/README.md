# Protein Complex Jigsaw (PCJ)

**From typed evidence to inspectable assembly, variant perturbation and misassembly.**

Protein Complex Jigsaw (PCJ) is an exact, inspectable framework for reasoning about protein-complex assembly from heterogeneous evidence. It separates evidence typing and provenance from model admission, enumerates admissible assembly mechanisms under declared constraints, and supports downstream perturbation analysis of variants or other molecular changes.

> **Naming note.** The project was previously distributed as `scijigsaw`. From v4.0.0 the preferred public software name is **Protein Complex Jigsaw**, the preferred Python namespace is `pcjigsaw`, and preferred command-line tools use the `pcjigsaw-` prefix. The historical `scijigsaw` namespace and commands remain available as compatibility aliases so existing analyses do not break.

## What PCJ does

PCJ represents evidence without silently converting one evidence type into another. The core model distinguishes physical contacts, temporal prerequisites, exclusions/competition and observed states or subcomplexes. It can then enumerate two complementary mechanism spaces:

- **seed-anchored assembly orders**, in which components join a growing seed-containing assembly;
- **binary merger trees**, in which independently formed subcomplexes can merge.

Exact queries include intermediate support, route admissibility, evidence/model diagnostics and minimum route-cover analyses.

## Variant perturbation and failure analysis

The `pcjigsaw.variant` module accepts an **explicitly declared molecular effect** — for example a component made unavailable, a lost interface, or a newly introduced exclusion — and recomputes the mechanism space. It reports mechanistic consequences including:

- whether the full productive target remains reachable;
- the largest reachable/trapped intermediate;
- first blocked transitions and their declared reasons;
- unavailable interfaces;
- whether any productive route survives.

PCJ does **not** infer the molecular consequence of a genomic variant from sequence alone. Variant-effect predictors, structural analysis or experiments should be used upstream to justify the perturbation supplied to PCJ.

## Misassembly is evidence-gated

Failure of productive assembly is **not** treated as evidence of aggregation. The `pcjigsaw.misassembly` layer evaluates off-pathway outcomes only when the user supplies an explicit rule with provenance. Thus a blocked productive route may be reported without any aggregation claim, while independently supported off-pathway interactions can be analysed separately.

PCJ does not infer pathogenicity, penetrance, tumour clonality, phylogeny or treatment response.

## Install

```bash
python -m pip install -e ".[dev]"
pytest -q
```

Preferred imports:

```python
from pcjigsaw import Assembly, VAMP2
from pcjigsaw import Perturbation, analyse_perturbation
```

Legacy imports continue to work:

```python
from scijigsaw import Assembly, VAMP2
```

## Command line

Existing deterministic tools are available under the new preferred prefix:

```bash
pcjigsaw-count all
pcjigsaw-extract examples/structures --contact-cutoff 5.0 --out interactions.csv
pcjigsaw-render examples/vamp2/proteins.csv examples/vamp2/interactions.csv --out board.svg
pcjigsaw-tiles examples/vamp2/proteins.csv examples/vamp2/interactions.csv --out kit.pdf
pcjigsaw-bench
```

The old `scijigsaw-*` spellings remain as compatibility aliases.

A declared perturbation can be analysed with:

```bash
pcjigsaw-variant vamp2 \
  --label "declared component loss" \
  --unavailable "Syntaxin-1A" \
  --provenance "experiment or upstream variant-effect analysis" \
  --json
```

## Reproduce the paper

The deterministic reproduction scripts remain unchanged:

```bash
python analysis/reproduce_all.py
python scripts/reproduce_numbers.py
python scripts/reproduce_figures.py
```

The repository also contains the structure-extraction benchmark, AlphaFold ensemble analyses, cryo-EM observation-set analyses, probabilistic extensions and the educational tile renderer used in the manuscript and supplement.

## Exactness and interpretation

Counts are exact **for the declared model**. They are not measured cellular pathway frequencies. Likewise, normalized branch shares in a weighted mechanism space are model-derived shares of remaining weighted completions, not direct estimates of biological population frequencies.

The physical/graphical jigsaw representation is an explanatory encoding of declared relations, not a molecular-shape prediction.

## Repository

Project repository: https://github.com/219plgit/protein-complex-jigsaw

Manuscript title:

**Protein Complex Jigsaw: from typed evidence to inspectable assembly, variant perturbation and misassembly**

## Citation

Please cite the manuscript and the versioned software release used in your analysis. See `CITATION.cff`.

## License

MIT License. See `LICENSE`.
