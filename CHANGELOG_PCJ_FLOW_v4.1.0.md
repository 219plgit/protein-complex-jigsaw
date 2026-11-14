# Protein Complex Jigsaw 4.1.0 - PCJ Flow core

## New

- `pcjigsaw.flow.ExactSeedFlow`
  - exact continuation flow over seed-anchored histories;
  - seeded additions must satisfy prerequisites/exclusions and contact the current seed-containing complex;
  - unit flow recovers the exact admissible history count for PCJ Flow encodings;
  - exact reward-proportional sampling.

- `pcjigsaw.flow.ExactMergerFlow`
  - exact flow over admissible unordered binary merger trees;
  - canonical root splits give each tree one recursive construction trajectory;
  - exact reward-normalised subcomplex support and tree sampling.

- Reward/diagnostic utilities
  - `plackett_luce_step_weight` for encounter-order priors with global-scale invariance;
  - `interface_potential` for merge-dependent structural weighting;
  - `effective_number` for entropy-equivalent mechanism counts;
  - `edge_additive_degeneracy` for fixed pairwise reward non-identifiability;
  - `compare_model_flows` for explicit separation of alternative hard models.

## Compatibility

The historical `scijigsaw` package remains installed. `Assembly.n_orders_permitted()` is not changed in this patch, preserving old prerequisite-only analyses. The manuscript-facing `pcjigsaw.flow.ExactSeedFlow` implements the stricter contact-connected seeded semantics required by Proposition 1.

## Regression gate

The supplied dedicated suite contains 11 tests and passes in the isolated verification harness. A release should additionally pass the repository-wide test suite after the patch is applied.

## Method → Tool → Education

- `education/vamp2_activity.py` regenerates the v21 VAMP2 activity from the research model:
  - the four-step figure;
  - the student sheet and answer key;
  - the printable tile sets;
  - three advanced PCJ Flow tasks: representation, reward and learned-sampler trust.
- Contacts, prerequisites, exclusions and queries keep their meaning: they become joins, must-wait cues, alternative occupancy and "can this state be built?".
- A model revision regenerates the research analysis and the answer key together. For example, the 1KIL revision changes 336 orders to 252.
- `education/pcj_teach.py` builds outreach and undergraduate packs for any small model. Ready-made packs: VAMP2, eIF3, chain, ring.
- `tests/test_teaching_pack.py` pins every answer key to the manuscript values and to the research counters.
- GFlowNet benchmark results are stored in `benchmark/gflownet/results/` and read by the advanced worksheet.

## Packaging: one executable model for research and teaching

- The research module `sj_prob.py` is now `pcjigsaw.prob` inside the installed package. The top-level `sj_prob.py` is an alias of the same module object, so existing scripts, notebooks and tests are unchanged.
- The educational code is the `pcjigsaw.education` subpackage, with the command `pcjigsaw-teach` (`vamp2`, `pack`, `check`; also `python -m pcjigsaw.education`).
- Package data ships with the release: the VAMP2 board, the typed VAMP2 encoding, the teaching models and the stored GFlowNet results. Tests require them to be byte-identical to `examples/vamp2` and `benchmark/gflownet/results`. The teaching eIF3 model must equal the research encoding.
- The built-in self-tests of `pcjigsaw.prob` now run under pytest. Two were updated to the seed-anchored contact rule:
  - the brute-force enumerator now requires a contact with the current complex;
  - the counterfactual example now drops a prerequisite that the contacts do not already imply.
- `CITATION.cff` version aligned with `pyproject.toml` (4.1.0).
- Verified from a wheel installed in a clean virtual environment: `pcjigsaw-teach check` reproduces every answer-key number, and the generated answer keys are byte-identical to those built from the source tree.

## Backward (disassembly) policies

- New module `pcjigsaw.backward`. The assembled complex can be taken as the single terminal of the seed-anchored construction, so states (sets of placed components) have several parents. A backward policy P_B ("which component dissociates next") then defines a distribution over assembly pathways, and the matching forward policy follows from detailed balance (Proposition 7).
- Built-in policies:
  - `uniform_backward`: no disassembly evidence;
  - `interface_backward`: weaker interfaces dissociate first;
  - `counting_backward`: recovers the unit-reward flow exactly.
- Admissibility is unchanged. Every admissible history keeps positive probability whenever P_B > 0 on admissible reverse steps.
- `tests/test_backward.py` checks:
  - the detailed-balance forward policy reproduces the backward-defined distribution;
  - counting recovers 1/6 and 1/252;
  - admissibility is invariant;
  - the eIF3 values reported in Supplementary Section S11 are reproduced.
- `benchmark/gflownet/run_seed_backward.py` trains a GFlowNet with fixed or learned backward policies on the 252-history VAMP2 state graph and compares it with exact flows. Results are in `benchmark/gflownet/results/seed_backward_results.json`.
- β-sensitivity and backward-entropy diagnostics:
  - `relation_count_backward` implements the minimal score b = n_C + n_P;
  - `backward_entropy` and `flow_weighted_backward_entropy` measure how many reverse explanations remain comparably plausible;
  - `beta_sensitivity` reports Δβ(Q), the range of a query's support over a declared β range.
- The tests pin:
  - Δβ = 0.50 for the eIF3 Prt1–Tif34 support under the interface score;
  - the relation-count score being uniform at every β once the redundant prerequisite Tif34≺Tif35 is removed.
- Results are in `benchmark/gflownet/results/eif3_beta_diagnostics.json`.
- Usefulness of evidence. β is read as the trust placed in the evidence summarised by the score b (0 ignore, small hint, large rule):
  - `evidence_usefulness` returns U(Q) = dS/dβ at β = 0 = −Cov₀(1_Q, D), where D sums, along a pathway, how much better anchored each dissociating component is than the average alternative; b is first divided by its typical spread (`evidence_scale`) so that different scores are comparable;
  - it also returns S0 (evidence ignored) and S_inf (evidence treated as a rule);
  - `interface_score`, `relation_count_score` and `score_backward` expose the scores used;
  - tests check the formula against finite differences and pin the reported eIF3 values (interface U = −0.75, S_inf = 0; relation count U = +0.48, S_inf = 1; U = 0 without the redundant prerequisite);
  - results are in `benchmark/gflownet/results/eif3_evidence_usefulness.json` (Supplementary Table S11).

## Forest (multi-parent) construction, contributed for v12

- `benchmark/gflownet/forest/` holds a bottom-up forest GFlowNet with the relation-count backward policy, P_B ∝ exp(-β b), b = n_C + n_P. It was contributed separately as a standalone "4.2.0" package and is integrated here as an experimental benchmark module rather than released as a separate package, which would have shadowed this distribution.
- β is fixed by default (learn_backward=False, fixed_beta=1.0). In this construction the end objects are merger trees, so β cannot change any subcomplex support, and a learned β is not identified (Supplementary Section S11.5).
- `tests/test_forest_backward.py` checks:
  - the per-tree sum of backward probabilities is 1 at β = 0, 1 and 4;
  - the contributed exact engine agrees with `pcjigsaw.prob.merger_F` (eIF3 14, SNARE 1,242);
  - the construction does produce states with several parents.
