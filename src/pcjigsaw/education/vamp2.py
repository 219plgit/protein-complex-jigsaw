#!/usr/bin/env python
"""VAMP2 activity: the v21 educational implementation, regenerated for PCJ Flow.

Method -> Tool -> Education. The activity is not a separate biological model.
Every relation shown to students is read from the same research objects used
in the paper, and every answer is computed from them:

  contact            -> permitted join           (examples/vamp2/interactions.csv)
  prerequisite       -> must-wait / support cue  (scijigsaw.cases.VAMP2.requires)
  exclusion          -> alternative occupancy    (partners sharing one site in the table)
  observation/query  -> "can this state be built?"
  model revision     -> regenerated pieces, cues and answer key

Outputs (OUTDIR):
  vamp2_steps.pdf/.png      four-step activity figure (Steps 1-4)
  vamp2_activity.pdf        student sheet for the four steps
  vamp2_answers.pdf         answer key, computed from the model
  advanced_worksheet.pdf    PCJ Flow tasks: representation, reward, GFlowNet
  advanced_answers.pdf      their answer key
  vamp2_answers.json        every number used
  kit_student.pdf / kit_teacher.pdf   the printable v21 tile sets

    pcjigsaw-teach vamp2 OUTDIR
"""
from __future__ import annotations

import json
import math
import os
import sys
from itertools import permutations

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

import pandas as pd                                                   # noqa: E402
import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages                  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Circle                 # noqa: E402

from scijigsaw.cases import VAMP2 as ASSEMBLY                         # noqa: E402
from scijigsaw.assembly import Assembly                               # noqa: E402
from pcjigsaw.prob import (TypedModel, seeded_F, calibrated_tree,          # noqa: E402
                     plackett_luce, n_eff, merger_F)
from .teach import Page, TeachingModel, compute, ACC, MUTED       # noqa: E402

BOARD = os.path.join(DATA, "vamp2_board")        # byte-identical to examples/vamp2 (tested)
SEED = ASSEMBLY.seed
FOOT = ("Protein Complex Jigsaw teaching pack · answers computed from the research model "
        "(scijigsaw.cases.VAMP2, examples/vamp2, pcjigsaw.prob)")


# =============================================================================
# 1. The model, read from the research objects
# =============================================================================
def board():
    prot = pd.read_csv(os.path.join(BOARD, "proteins.csv"))
    inter = pd.read_csv(os.path.join(BOARD, "interactions.csv"))
    return prot, inter


def contacts(inter):
    return {frozenset((a, b)) for a, b in zip(inter.protein_a, inter.protein_b)}


def site_alternatives(inter, host=SEED):
    """Partners that bind the SAME declared site on `host` occupy it in turn:
    this is how exclusion becomes alternative occupancy on the board."""
    by_site = {}
    for a, b, sa, sb in zip(inter.protein_a, inter.protein_b, inter.site_on_a, inter.site_on_b):
        if a == host:
            by_site.setdefault(sa, []).append(b)
        elif b == host:
            by_site.setdefault(sb, []).append(a)
    return {s: sorted(ps) for s, ps in by_site.items() if len(ps) > 1}


def exclusions(inter):
    ex = set()
    for ps in site_alternatives(inter).values():
        for i, p in enumerate(ps):
            for q in ps[i + 1:]:
                ex.add(frozenset((p, q)))
    # every exclusion declared in the research Assembly must be reproduced
    for A, B in ASSEMBLY.excludes:
        for a in A:
            for b in B:
                assert frozenset((a, b)) in ex, (a, b)
    return ex


def requires_for(p, inter):
    """Seating requirement of p: the research prerequisites; board-only
    alternatives (not in the fusion target) need only their declared host."""
    if p in ASSEMBLY.requires:
        return set(ASSEMBLY.requires[p])
    return {q for e in contacts(inter) if p in e for q in e if q != p}


def buildable(state, inter):
    """Can this state be built? Grow from the seed inside `state`, adding a
    component only when its requirements are present, with no two mutually
    exclusive components in the state."""
    state = set(state)
    ex = exclusions(inter)
    for e in ex:
        if e <= state:
            return False, "two components compete for the same site: " + " / ".join(sorted(e))
    if SEED not in state:
        return False, f"every state on this board grows from {SEED}"
    S = {SEED}
    changed = True
    while changed:
        changed = False
        for p in sorted(state - S):
            if requires_for(p, inter) <= S:
                S.add(p); changed = True
    if S == state:
        return True, "buildable"
    reasons = []
    for p in sorted(state - S):
        absent = sorted(requires_for(p, inter) - state)
        if absent:
            reasons.append(f"{p} needs {' + '.join(absent)}, which is not in the state")
        else:
            reasons.append(f"{p} waits for " + " + ".join(sorted(requires_for(p, inter) - S)))
    why = "; ".join(reasons)
    return False, why


def target_units():
    return [SEED] + sorted(ASSEMBLY.requires)


def seeded_orders(requires):
    units = [u for u in target_units() if u != SEED]
    out = []

    def grow(S, order):
        if len(order) == len(units):
            out.append(tuple(order)); return
        for p in units:
            if p not in S and requires[p] <= S:
                grow(S | {p}, order + [p])
    grow({SEED}, [])
    return out


def typed_seating_model():
    """The paper's 252-history VAMP2 seating model as a TypedModel (stored in
    models/vamp2_typed.json). Its contacts must equal the board's contacts
    within the fusion target, and its seed-anchored count must equal the
    research Assembly's."""
    spec = json.load(open(os.path.join(DATA, "models", "vamp2_typed.json")))
    C = frozenset(frozenset(e) for e in spec["contacts"])
    _, inter = board()
    assert C == {e for e in contacts(inter) if e <= set(target_units())}, "board and typed model disagree"
    V = tuple(sorted({u for e in spec["contacts"] for u in e}))
    return TypedModel(V=V, C=C, P=tuple(tuple(x) for x in spec["prerequisites"]))


# =============================================================================
# 2. All numbers
# =============================================================================
def compute_vamp2():
    prot, inter = board()
    con = contacts(inter)
    req = {p: set(r) for p, r in ASSEMBLY.requires.items()}
    orders = seeded_orders(req)
    n_units = len(target_units()) - 1
    # 1KIL revision: before re-extraction Complexin was encoded on VAMP2 + SNAP25
    req_before = dict(req); req_before["Complexin"] = {"VAMP2", "SNAP25"}
    before = len(seeded_orders(req_before))
    tm = typed_seating_model()
    R = {
        "seed": SEED,
        "fusion_target": target_units(),
        "core": ["SNAP25", "Syntaxin-1A", "VAMP2"],
        "core_contacts": sorted("–".join(sorted(e)) for e in con if e <= {"SNAP25", "Syntaxin-1A", "VAMP2"}),
        "bridges": {p: sorted(req[p]) for p in ("Syt-1", "Complexin")},
        "syt1_complexin_contact": frozenset(("Syt-1", "Complexin")) in con,
        "accessories": {p: sorted(req[p]) for p in ("Munc18-1", "SNCA", "Synaptophysin")},
        "alternatives": site_alternatives(inter),
        "orders_total": math.factorial(n_units),
        "orders_admissible": len(orders),
        "orders_admissible_assembly": ASSEMBLY.n_orders_permitted(contact_connected=True),
        "orders_admissible_typed": seeded_F(tm, SEED)(frozenset()),
        "merger_trees_typed": merger_F(tm)(frozenset(tm.V)),
        "revision_1KIL": {"before": before, "after": len(orders),
                          "complexin_before": ["VAMP2", "SNAP25"], "complexin_after": sorted(req["Complexin"])},
        "core_first_orders": sum(set(o[:2]) == {"SNAP25", "Syntaxin-1A"} for o in orders),
        "syntaxin_requires": sorted(req["Syntaxin-1A"]),
        "core_orders": [list(o) for o in permutations(["SNAP25", "Syntaxin-1A"])
                        if all(req[p] <= {SEED} | set(o[:i]) for i, p in enumerate(o))],
        "queries": [],
    }
    assert R["orders_admissible"] == R["orders_admissible_assembly"] == R["orders_admissible_typed"] == 252
    assert R["merger_trees_typed"] == 536
    Q = [["VAMP2", "SNAP25", "Syntaxin-1A"],
         ["VAMP2", "Complexin"],
         ["VAMP2", "SNAP25", "Syntaxin-1A", "Syt-1", "Complexin"],
         ["VAMP2", "AP180"],
         ["VAMP2", "SNAP25", "AP180"],
         ["VAMP2", "AP180", "CALM"],
         ["VAMP2", "Syntaxin-1A", "Munc18-1"]]
    for q in Q:
        ok, why = buildable(q, inter)
        sup = None
        if ok and set(q) <= set(target_units()):
            sup = sum(any(set((SEED,) + o[:k]) == set(q) for k in range(len(o) + 1)) for o in orders)
        R["queries"].append({"state": q, "buildable": ok, "why": why,
                             "histories_through_state": sup})
    # reward task. The only N_eff value shown to students is the GTEx result
    # REPORTED in the main text (it needs the GTEx matrices, which are not
    # bundled). The code verifies, on the research model, the two properties
    # the exercise asks about, without introducing new numerical claims:
    #   (i) every positive Plackett-Luce prior keeps all histories admissible;
    #  (ii) multiplying every abundance by one factor leaves the distribution unchanged.
    import random
    from pcjigsaw.prob import leaf_masses
    rng = random.Random(2026)
    checks = []
    for _ in range(5):
        ab = {v: rng.uniform(0.1, 100.0) for v in tm.V}
        r1 = calibrated_tree(tm, SEED, step_weight=plackett_luce(tm, SEED, ab))
        r2 = calibrated_tree(tm, SEED, step_weight=plackett_luce(tm, SEED, {v: 10 * a for v, a in ab.items()}))
        m1 = dict(leaf_masses(r1)); m2 = dict(leaf_masses(r2))
        checks.append({"admissible": sum(m > 0 for m in m1.values()),
                       "rescaling_max_change": max(abs(m1[k] - m2[k]) for k in m1)})
    R["reward"] = {
        "reported_gtex": {"histories_admissible": 252, "N_eff_v10": "57.8 ± 18.4", "N_eff_v8": "63.7 ± 25.2",
                          "source": "main text (GTEx brain tissues; mean ± SD across tissues)"},
        "admissible_under_all_checked_priors": all(c["admissible"] == R["orders_admissible"] for c in checks),
        "rescaling_invariant": all(c["rescaling_max_change"] < 1e-12 for c in checks),
        "checks": checks,
    }
    return R


def compute_gflownet():
    d = os.path.join(DATA, "gflownet")             # byte-identical to benchmark/gflownet/results (tested)
    S = json.load(open(os.path.join(d, "snare_results.json")))
    L = json.load(open(os.path.join(d, "large_results.json")))
    LL = json.load(open(os.path.join(d, "large_long_10000_0.3.json")))
    rug = "rugged merge energies, sigma=1.5"

    def rng(xs, f="{:.2g}"):
        return f.format(min(xs)) + "–" + f.format(max(xs))
    rows = [
        ["SNARE (1,242)", "uniform", "TV " + rng([r["TV"] for r in S["uniform"]]), "trust"],
        ["SNARE (1,242)", "rugged", "TV " + rng([r["TV"] for r in S[rug]], "{:.3f}"), "trust"],
        ["synthetic (3.6e8)", "uniform", "KL " + rng([r["KL_q_p"] for r in L["uniform"]["runs"]], "{:.3f}"), "trust"],
        ["synthetic (3.6e8)", "rugged", "log Z short by " + rng([L[rug]["log_trees_or_Z"] - r["logZ_learned"] for r in L[rug]["runs"]], "{:.1f}")
         + " nats; max error " + rng([r["support_maxAE"] for r in L[rug]["runs"]]), "do not trust"],
        ["synthetic (3.6e8)", "rugged, 10k steps", "log Z short by " + rng([r["logZ_exact"] - r["logZ_learned"] for r in LL], "{:.2f}")
         + " nats; max error " + rng([r["support_maxAE"] for r in LL]), "do not trust"],
    ]
    return {"rows": rows}


# =============================================================================
# 3. Four-step activity figure, drawn from the model
# =============================================================================
FILL = {"fusion": "#dfe3ea", "regulation": "#dbe7f7", "chaperone": "#fdeab0", "retrieval": "#fff4e0"}
EDGEC = {"fusion": "#4b5563", "regulation": "#3b6fb6", "chaperone": "#b08900", "retrieval": "#d97706"}


def draw_steps(R, out_pdf, out_png):
    prot, inter = board()
    fn = dict(zip(prot.name, prot.function))
    fig, ax = plt.subplots(figsize=(7.2, 9.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12.6); ax.set_axis_off()

    def tile(x, y, w, h, name, dashed=False):
        f = fn.get(name, "regulation")
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=FILL[f], ec=EDGEC[f], lw=1.0, ls=(0, (3, 2)) if dashed else "-", zorder=2))
        ax.text(x + w / 2, y + h / 2, name, ha="center", va="center", fontsize=7.2, fontweight="bold", zorder=3)

    def dot(x, y):
        ax.add_patch(Circle((x, y), 0.09, fc="white", ec="#4b5563", lw=0.9, zorder=4))

    W, H = 1.7, 0.75
    X = {"SNAP25": 0.6, "Syntaxin-1A": 0.6 + W, "VAMP2": 0.6 + 2 * W}

    def core(y):
        for p, x in X.items():
            tile(x, y, W, H, p)
        dot(X["Syntaxin-1A"], y + H / 2); dot(X["VAMP2"], y + H / 2)

    def bridges(y):
        top = y + H
        a, b = X["Syntaxin-1A"], X["VAMP2"]
        tile(a - W / 2 + 0.03, top, W - 0.1, 0.6, "Syt-1")
        tile(b - W / 2 + 0.07, top, W - 0.1, 0.6, "Complexin")
        for x in (a - W / 4, a + W / 4 - 0.05, b - W / 4 + 0.1, b + W / 4):
            dot(x, top)

    def accessories(y):
        tile(X["Syntaxin-1A"] + 0.25, y - 0.6, 1.2, 0.6, "Munc18-1"); dot(X["Syntaxin-1A"] + 0.85, y)
        tile(X["VAMP2"] + 0.05, y - 0.6, 0.75, 0.6, "SNCA"); dot(X["VAMP2"] + 0.42, y)
        tile(X["VAMP2"] + 0.88, y - 0.6, 1.45, 0.6, "Synaptophysin"); dot(X["VAMP2"] + 1.5, y)

    def head(y, s):
        ax.text(0.3, y, s, fontsize=9.5, fontweight="bold", va="center")

    def note(y, s):
        ax.text(0.3, y, s, fontsize=6.8, color="#374151", va="top", linespacing=1.35)

    br = R["bridges"]
    # Step 1
    head(12.3, "Step 1 · discover contacts and close the core")
    core(11.0)
    note(10.75, f"SNAP25, Syntaxin-1A and VAMP2 form the core; all three pairs are declared contacts "
         f"({', '.join(R['core_contacts'])}).\nConnectors show which pieces may join, not molecular shape or affinity.")
    # Step 2
    head(9.75, "Step 2 · discover precedence: seat the bridges")
    core(8.0); bridges(8.0)
    note(7.75, f"Syt-1 needs {' + '.join(br['Syt-1'])}; Complexin needs {' + '.join(br['Complexin'])}. A bridge cannot be seated "
         "before both supports are present.\n"
         + ("No direct Syt-1–Complexin contact is declared: the two bridges do not join each other."
            if not R["syt1_complexin_contact"] else "Syt-1 and Complexin are in contact."))
    # Step 3
    head(6.75, "Step 3 · discover exclusion: alternative occupancy")
    core(4.7); bridges(4.7); accessories(4.7)
    alts = sorted(list(R["alternatives"].values())[0], key=lambda p: (p != "SNAP25", p))
    bx, by = 6.75, 3.95
    ax.add_patch(FancyBboxPatch((bx, by), 3.05, 2.35, boxstyle="round,pad=0,rounding_size=0.1",
                                fc="#fffbeb", ec="#d97706", lw=0.9, ls=(0, (3, 2))))
    ax.text(bx + 1.52, by + 2.05, "one VAMP2 SNARE-motif site", ha="center", fontsize=7, fontweight="bold", color="#92400e")
    ax.text(bx + 1.52, by + 1.75, "choose one occupant at a time", ha="center", fontsize=6.5, color="#92400e")
    for i, p in enumerate(alts):
        tile(bx + 0.12 + i * 0.97, by + 0.95, 0.88, 0.5, p, dashed=(p != "SNAP25"))
    ax.text(bx + 1.52, by + 0.45, "only SNAP25 leads to the\nfusion-compatible state", ha="center", fontsize=6.3, color="#92400e")
    note(3.85, "Accessories join their declared partners: Munc18-1 on Syntaxin-1A; SNCA and Synaptophysin on VAMP2.\n"
         f"{', '.join(alts[:-1])} and {alts[-1]} bind the same declared VAMP2 site, so they are alternatives, never simultaneous.")
    # Step 4
    head(2.85, "Step 4 · connect the pieces to the exact count")
    rv = R["revision_1KIL"]
    ax.add_patch(FancyBboxPatch((0.4, 0.25), 9.3, 2.25, boxstyle="round,pad=0,rounding_size=0.1",
                                fc="#f9fafb", ec="#d1d5db", lw=0.8))
    ax.text(0.7, 2.2, f"{R['orders_admissible']}", fontsize=20, fontweight="bold", color=ACC, va="top")
    ax.text(2.0, 2.15, f"admissible seed-anchored orders of the eight-component fusion state,\n"
            f"out of {R['orders_total']:,} unconstrained orders after {SEED}.", fontsize=7.2, va="top")
    ax.text(0.7, 1.15, f"{rv['before']} → {rv['after']}", fontsize=13, fontweight="bold", color="#b91c1c", va="top")
    ax.text(3.0, 1.15, f"Revision: re-extracting PDB 1KIL moves Complexin's support\n"
            f"from {' + '.join(rv['complexin_before'])} to {' + '.join(rv['complexin_after'])}. Pieces, cues and\n"
            "answer key are regenerated from the model, not edited by hand.", fontsize=7.2, va="top")
    fig.savefig(out_pdf, bbox_inches="tight"); fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 4. Sheets
# =============================================================================
def activity_sheets(R, out_ws, out_ans):
    for out, ans in ((out_ws, False), (out_ans, True)):
        with PdfPages(out) as pdf:
            p = Page(pdf, "The VAMP2 jigsaw" + (": answer key" if ans else ""),
                     "Four steps · construct, test, explain, revise", FOOT)
            p.para("You will build the fusion-compatible VAMP2 complex from printed tiles. The connectors are keys: they show "
                   "which proteins may join according to the declared evidence, not the shape of the molecules. "
                   "Work in pairs and write down your reasons.", size=9.2)
            p.heading("Step 1 · Contacts and the core")
            p.para("Build SNAP25 + Syntaxin-1A + VAMP2, starting from VAMP2. In how many orders can the core be built? "
                   "Which pairs of the core are in contact?")
            if ans:
                n_core = len(R["core_orders"])
                p.para(f"Answer: {n_core} order{'s' if n_core != 1 else ''} ("
                       + "; ".join(" → ".join([SEED] + o) for o in R["core_orders"])
                       + f"), because Syntaxin-1A needs {' and '.join(R['syntaxin_requires'])}. Declared core contacts: "
                       f"{', '.join(R['core_contacts'])}.", col=ACC)
            else:
                p.lines(2)
            p.heading("Step 2 · Precedence")
            p.para("Try to seat Syt-1 and Complexin. When is the earliest moment each can be placed? Do they touch each other?")
            if ans:
                b = R["bridges"]
                p.para(f"Answer: Syt-1 needs {' + '.join(b['Syt-1'])}; Complexin needs {' + '.join(b['Complexin'])}; "
                       "both can be seated only after Syntaxin-1A. "
                       + ("No Syt-1–Complexin contact is declared." if not R["syt1_complexin_contact"] else ""), col=ACC)
            else:
                p.lines(2)
            p.heading("Step 3 · Exclusion and alternative occupancy")
            p.para("Which tiles compete for the same VAMP2 socket? Can each of these states be built?")
            for q in R["queries"]:
                line = "   {" + ", ".join(q["state"]) + "}"
                if ans:
                    line += ("   yes" if q["buildable"] else "   no — " + q["why"])
                    if q["histories_through_state"] is not None:
                        line += f"  ({q['histories_through_state']} of {R['orders_admissible']} histories pass through it)"
                p.para(line, size=8.6, col=ACC if ans else "#1f2328", gap=0.002)
            if not ans:
                p.lines(1)
            p.heading("Step 4 · The exact count and a revision")
            p.para(f"Rebuild the full eight-component state in a different valid order. The software finds every valid order: "
                   f"how many are there out of {R['orders_total']:,}? Now move Complexin's support from Syntaxin-1A to SNAP25, as in "
                   "the evidence before PDB 1KIL was re-extracted. Does the count go up or down, and why?")
            if ans:
                rv = R["revision_1KIL"]
                p.para(f"Answer: {R['orders_admissible']} of {R['orders_total']:,}. With Complexin on {' + '.join(rv['complexin_before'])} "
                       f"there are {rv['before']}: Complexin can then be seated as soon as SNAP25 is present, before Syntaxin-1A, "
                       "so more orders are allowed. The structure removed that freedom.", col=ACC)
                p.para("Possible ≠ frequent: these counts say what the declared model permits, not what happens most often in a "
                       "synapse.", size=8.6, col=MUTED)
            else:
                p.lines(2)
            p.close()


def advanced_sheets(R, Re, G, out_ws, out_ans):
    for out, ans in ((out_ws, False), (out_ans, True)):
        with PdfPages(out) as pdf:
            p = Page(pdf, "From pieces to flow" + (": answer key" if ans else ""),
                     "Advanced tasks for PCJ Flow (final-year undergraduate)", FOOT)
            p.heading("A · Representation (yeast eIF3)")
            p.para("Native mass spectrometry detects a Tif34–Tif35 dimer. With Prt1 as the seed and one component added at a "
                   "time, can this dimer ever be an intermediate? What changes if separately built subcomplexes may join?")
            if ans:
                sq, nq = Re["merger"]["query_support"]
                p.para(f"Answer: seed-anchored — {Re['seeded']['query_support'][0]} of {Re['seeded']['count']} histories, because every "
                       f"intermediate contains Prt1. Merger trees — {sq} of {nq}. The evidence is unchanged; the representation "
                       "(the rule for what counts as a step) decides whether the dimer is possible.", col=ACC)
            else:
                p.lines(3)
            g = R["reward"]["reported_gtex"]
            p.heading("B · Reward: redistribute, never remove")
            p.para("An encounter-order prior weights each step by the abundance of the added protein divided by the total abundance "
                   "of the proteins that could be added at that moment (Plackett–Luce). Using GTEx brain-tissue expression as a proxy "
                   f"for abundance, the paper reports that the effective number of VAMP2 histories, N_eff = 2^H, falls from "
                   f"{R['orders_admissible']} to {g['N_eff_v10']} (GTEx v10, mean ± SD across tissues). How many of the "
                   f"{R['orders_admissible']} histories are still possible? Why does N_eff fall? What would happen if every abundance "
                   "in a tissue were multiplied by ten?")
            if ans:
                ok_a, ok_s = R["reward"]["admissible_under_all_checked_priors"], R["reward"]["rescaling_invariant"]
                assert ok_a and ok_s
                p.para(f"Answer: all {R['orders_admissible']} remain possible, because every weight is positive and a reward can "
                       "only redistribute probability among admissible histories (admissibility is reward-invariant). N_eff falls "
                       "because the prior concentrates probability on some histories. Multiplying every abundance by the same factor "
                       "changes nothing: the factor cancels in each step's ratio (scale invariance), so only relative abundances "
                       "carry information about order. Both properties are verified on the research model by the code that "
                       "generates this sheet.", col=ACC)
            else:
                p.lines(4)
            p.heading("C · When can a learned sampler be trusted?")
            p.para("A GFlowNet learns to sample assembly trees instead of counting them. The exact flow is the reference. "
                   "For each benchmark below, decide whether the learned sampler can be trusted, and say what you would check.")
            p.table(["space", "reward", "learned vs exact" if ans else "learned vs exact", "verdict" if ans else ""],
                    [r[:3] + ([r[3]] if ans else [""]) for r in G["rows"]], [0.16, 0.14, 0.43, 0.09], size=7.6)
            if ans:
                p.para("Answer: on the small SNARE space and with unit reward the sampler matches the exact flow. On the large "
                       "rugged space it under-estimates the partition function and misses supports by up to about half; longer "
                       "training reduces but does not remove the gap. A learned sampler is trustworthy only where it has been "
                       "checked against an exact reference or an independent diagnostic.", col=ACC)
            else:
                p.lines(3)
            p.close()


def build(out_dir, kits=True):
    os.makedirs(out_dir, exist_ok=True)
    R = compute_vamp2()
    Re = compute(TeachingModel.from_dir(os.path.join(DATA, "models", "eif3")))
    G = compute_gflownet()
    with open(os.path.join(out_dir, "vamp2_answers.json"), "w") as fh:
        json.dump({"vamp2": R, "eif3": Re, "gflownet": G}, fh, indent=1, default=list)
    draw_steps(R, os.path.join(out_dir, "vamp2_steps.pdf"), os.path.join(out_dir, "vamp2_steps.png"))
    activity_sheets(R, os.path.join(out_dir, "vamp2_activity.pdf"), os.path.join(out_dir, "vamp2_answers.pdf"))
    advanced_sheets(R, Re, G, os.path.join(out_dir, "advanced_worksheet.pdf"), os.path.join(out_dir, "advanced_answers.pdf"))
    if kits:
        from scijigsaw.tiles import tiles
        tiles(os.path.join(BOARD, "proteins.csv"), os.path.join(BOARD, "interactions.csv"),
              os.path.join(out_dir, "kit.pdf"), title="VAMP2 board")
    return R
