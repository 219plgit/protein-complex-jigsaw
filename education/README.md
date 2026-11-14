# Protein Complex Jigsaw: Method → Tool → Education

The educational code is part of the installed package (`pcjigsaw.education`),
next to the research code it compiles from (`scijigsaw.cases`, `pcjigsaw.prob`,
`scijigsaw.tiles`). After `pip install protein-complex-jigsaw`:

```
pcjigsaw-teach check                  # recompute every answer-key number and compare with the paper
pcjigsaw-teach vamp2 vamp2_pack       # the VAMP2 activity, tile sets and advanced tasks
pcjigsaw-teach pack eif3 eif3_pack    # outreach + undergraduate pack for a small model
```

The VAMP2 board, the typed VAMP2 encoding, the teaching models and the stored
GFlowNet results ship as package data. Tests check that they are byte-identical
to `examples/vamp2` and `benchmark/gflownet/results`, so there is one model, not
a teaching copy. The scripts in this folder are thin wrappers around the package.

The teaching material is not a separate biological model. It renders selected
relations of the **same typed model** used for research, keeping their meaning:

| Relation in the model | Research tool (PCJ Flow) | Student activity |
|---|---|---|
| contact (u, v) ∈ C | tests whether two components or blocks may join | matching connectors: a permitted join |
| prerequisite u ≺ v | tests whether one component must be present first | a bridge / must-wait cue that cannot be seated too early |
| exclusion u ⊥ v | forbids incompatible co-occupancy | alternative occupancy: pieces compete for one socket |
| observation / query B | support of B in the admissible space | "can this state be built?" |
| model revision | recomputes the mechanism space and its flow | regenerates the pieces, the cues and the answer key |

- **Method:** typed evidence, explicit admission, a chosen assembly representation, and exact flow.
- **Tool:** makes the same relations executable, inspectable and revisable.
- **Education:** exposes those relations as construction, prediction, counterfactual and explanation tasks.

Students construct, test, explain and revise; they do not reproduce the professional computation.

**Every answer is computed from the model.** When the evidence changes, the
research analysis and the answer key are regenerated together; nothing is
edited by hand. `tests/test_teaching_pack.py` pins the answer keys to the
paper:
- VAMP2: 252 of 5,040 orders, 536 merger trees, and 336 → 252 after the 1KIL revision;
- eIF3: 6 histories, 14 trees, and Tif34–Tif35 in 5 of 14.

This establishes semantic continuity between the research and educational
representations. It does not establish learning gains, which would need a
separate prospective study.

## 1. The VAMP2 activity (main implementation)

```
pcjigsaw-teach vamp2 OUTDIR          # or: python -m pcjigsaw.education vamp2 OUTDIR
```

The activity has four steps, using the board in `examples/vamp2` and the research model in
`scijigsaw.cases.VAMP2`:

1. **Contacts and the core.** Close SNAP25 + Syntaxin-1A + VAMP2. All three pairs are declared contacts.
2. **Precedence.** Seat the bridges:
   - Syt-1 needs SNAP25 + Syntaxin-1A;
   - Complexin needs Syntaxin-1A + VAMP2;
   - no Syt-1–Complexin contact is declared.
3. **Exclusion.** SNAP25, AP180 and CALM bind the same declared VAMP2 site, so they are alternatives and never simultaneous.
4. **Exact count and revision.**
   - 252 of 5,040 orders are admissible.
   - Re-extracting PDB 1KIL moved Complexin from VAMP2 + SNAP25 to Syntaxin-1A + VAMP2, changing 336 to 252.

Outputs:
- `vamp2_steps.pdf` — the four-step figure, drawn from the model;
- `vamp2_activity.pdf` and `vamp2_answers.pdf` — student sheet and answer key;
- `kit_student.pdf` and `kit_teacher.pdf` — the printable tile sets, with identical cut geometry;
- `advanced_worksheet.pdf` and `advanced_answers.pdf` — three PCJ Flow tasks:
  - **A. Representation:** why eIF3 Tif34–Tif35 has support 0/6 seed-anchored but 5/14 in merger trees;
  - **B. Reward:** using the reported GTEx result (252 histories admissible, N_eff 57.8 ± 18.4), why admissibility is unchanged, why N_eff falls, and why rescaling all abundances changes nothing;
  - **C. Learned samplers:** when an amortised GFlowNet can be trusted, judged against exact flow using `benchmark/gflownet/results`;
- `vamp2_answers.json` — every number used.

Task B introduces no new numbers: it quotes the GTEx result from the paper, and the generator verifies on the research model that positive priors keep all 252 histories and that global rescaling is exact.

## 2. Packs for any small model

```
pcjigsaw-teach pack eif3 OUTDIR       # built-in: eif3, chain4, cycle4; or a directory
```

From `proteins.csv` and `interactions.csv` (plus an optional `model.json` with
`seed` and `query`), this writes:
- tile kits;
- a school outreach card (grow from one piece, class count, change one rule, break a connection) with facilitator notes;
- an undergraduate worksheet with an answer key: seed-anchored vs merger counts, perturbation, uniform-step vs exact-flow probabilities, and chain vs ring scoring.

Teaching models contain contacts only. Keep them to about eight components.

Ready-made packs (regenerated by the commands above) are in `packs/`:
- `vamp2`
- `eif3`
- `chain4`
- `cycle4`

## Notes for teachers

- **Possible is not frequent.** Counts and supports describe what the declared model permits, not how often something happens in a cell.
- **Connectors are keys, not molecular surfaces.** The activity samples admissible orders, not kinetics.
- **Placeholder values in the eIF3 tiles.** The `kd_nM` and site columns in the eIF3 teaching CSV are placeholders the tile renderer needs; no count uses them.
