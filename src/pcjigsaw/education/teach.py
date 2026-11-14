#!/usr/bin/env python
"""Protein Complex Jigsaw -- teaching-pack generator.

From one small contact model (the same proteins.csv / interactions.csv used by
the tile renderer) this script produces a complete, self-checking teaching pack:

    kit_student.pdf / kit_teacher.pdf   printable cut-out tiles (scijigsaw.tiles)
    outreach_card.pdf                   school activity + facilitator notes
    worksheet.pdf                       undergraduate worksheet (no answers)
    answers.pdf                         undergraduate answer key
    answers.json                        every number used, machine-readable

Every number printed in a worksheet or answer key is computed here by exact
enumeration; nothing is typed by hand. This is the teaching counterpart of the
paper's rule that every graphic is either an exact result or clearly schematic.

The enumerator is deliberately small and self-contained so that it can be read
by students. It implements exactly two of the paper's representations:

  seed-anchored  start from the seed; each step adds ONE component that contacts
                 the current complex (Proposition 1's contact rule)
  merger         binary trees; each merge joins two connected blocks that touch
                 through at least one declared contact; trees are counted once
                 (canonical splits)

Teaching models contain contacts only (no prerequisites or exclusions), and are
intended for up to about eight components.

Usage
    pcjigsaw-teach pack eif3 OUTDIR
    pcjigsaw-teach pack MY_MODEL_DIR OUTDIR --seed X --query Y,Z
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, permutations

import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BUILTIN_MODELS = ("eif3", "chain4", "cycle4")


def model_path(name_or_dir):
    """A built-in teaching model name (eif3, chain4, cycle4) or a directory."""
    if name_or_dir in BUILTIN_MODELS:
        return os.path.join(DATA, "models", name_or_dir)
    return name_or_dir


# =============================================================================
# 1. The model
# =============================================================================
class TeachingModel:
    def __init__(self, components, contacts, seed, name="model", query=None, source=""):
        self.components = list(components)
        self.contacts = [tuple(sorted(e)) for e in contacts]
        self.seed = seed
        self.name = name
        self.source = source
        if seed not in self.components:
            raise ValueError(f"seed {seed!r} is not a component")
        for u, v in self.contacts:
            if u not in self.components or v not in self.components:
                raise ValueError(f"contact {u}-{v} uses an unknown component")
        self.query = tuple(query) if query else self._default_query()

    @classmethod
    def from_dir(cls, path, seed=None, query=None):
        prot = pd.read_csv(os.path.join(path, "proteins.csv"))
        inter = pd.read_csv(os.path.join(path, "interactions.csv"))
        meta = {}
        mj = os.path.join(path, "model.json")
        if os.path.exists(mj):
            with open(mj) as fh:
                meta = json.load(fh)
        return cls(prot["name"].tolist(),
                   list(zip(inter["protein_a"], inter["protein_b"])),
                   seed or meta.get("seed") or prot["name"].iloc[0],
                   name=meta.get("name", os.path.basename(path.rstrip("/"))),
                   query=query or meta.get("query"),
                   source=meta.get("source", ""))

    # --- helpers -------------------------------------------------------------
    def without(self, contact):
        c = tuple(sorted(contact))
        return TeachingModel(self.components, [e for e in self.contacts if e != c], self.seed,
                             self.name, self.query, self.source)

    def touches(self, A, B):
        return [e for e in self.contacts if (e[0] in A and e[1] in B) or (e[1] in A and e[0] in B)]

    def connected(self, S):
        S = set(S)
        if not S:
            return False
        start = next(iter(S)); seen = {start}; stack = [start]
        while stack:
            u = stack.pop()
            for a, b in self.contacts:
                for x, y in ((a, b), (b, a)):
                    if x == u and y in S and y not in seen:
                        seen.add(y); stack.append(y)
        return seen == S

    def n_cycles(self):
        """Independent cycles of the contact graph: |E| - |V| + #components."""
        comps, seen = 0, set()
        for v in self.components:
            if v not in seen:
                comps += 1
                stack = [v]; seen.add(v)
                while stack:
                    u = stack.pop()
                    for a, b in self.contacts:
                        for x, y in ((a, b), (b, a)):
                            if x == u and y not in seen:
                                seen.add(y); stack.append(y)
        return len(self.contacts) - len(self.components) + comps

    def _default_query(self):
        off = [e for e in self.contacts if self.seed not in e]
        return off[0] if off else self.contacts[0]

    # =========================================================================
    # 2. Seed-anchored histories
    # =========================================================================
    def seeded_histories(self):
        out = []

        def grow(S, order):
            if len(S) == len(self.components):
                out.append(tuple(order)); return
            for c in self.components:
                if c not in S and self.touches({c}, S):
                    grow(S | {c}, order + [c])
        grow({self.seed}, [self.seed])
        return out

    def naive_step_probability(self, history):
        """Probability of `history` if a builder picks uniformly among the legal
        next components at every step (no look-ahead)."""
        S, p = {history[0]}, Fraction(1)
        for c in history[1:]:
            legal = [x for x in self.components if x not in S and self.touches({x}, S)]
            p *= Fraction(1, len(legal))
            S.add(c)
        return p

    def flow_step_probability(self, history):
        """Same history under the exact-flow policy P_F = F(child)/F(state)."""
        @lru_cache(None)
        def F(S):
            if len(S) == len(self.components):
                return 1
            return sum(F(S | {x}) for x in self.components if x not in S and self.touches({x}, S))
        S, p = frozenset({history[0]}), Fraction(1)
        for c in history[1:]:
            child = S | {c}
            p *= Fraction(F(child), F(S))
            S = child
        return p

    # =========================================================================
    # 3. Merger trees (canonical splits; each tree counted once)
    # =========================================================================
    def merger_trees(self):
        """Each tree is a tuple of merges (A, B, crossed_contacts), leaves to root."""
        conn = lru_cache(None)(lambda S: self.connected(S))

        @lru_cache(None)
        def T(S):
            if len(S) == 1:
                return [()]
            out = []
            m = min(S); rest = sorted(S - {m})
            for k in range(len(rest)):
                for c in combinations(rest, k):
                    A = frozenset((m,) + c); B = S - A
                    if B and conn(A) and conn(B):
                        cross = tuple(sorted(self.touches(A, B)))
                        for ta in T(A):
                            for tb in T(B):
                                out.append(ta + tb + ((A, B, cross),))
            return out
        if not self.connected(self.components):
            return []
        return T(frozenset(self.components))

    @staticmethod
    def blocks(tree):
        return {A | B for A, B, _ in tree}


# =============================================================================
# 4. All numbers used by the pack
# =============================================================================
def compute(model: TeachingModel):
    H = model.seeded_histories()
    T = model.merger_trees()
    q = frozenset(model.query)
    n = len(model.components)
    res = {
        "model": model.name, "source": model.source, "seed": model.seed,
        "components": model.components, "contacts": [list(e) for e in model.contacts],
        "independent_cycles": model.n_cycles(),
        "query": list(model.query),
        "all_orders_from_seed": math.factorial(n - 1),
        "seeded": {"count": len(H), "histories": [list(h) for h in H]},
        "merger": {"count": len(T)},
    }
    # query support
    # In a seed-anchored history the intermediates are the prefixes; q is an
    # intermediate iff some prefix equals q.
    sup_seed = sum(any(set(h[:k]) == q for k in range(1, n + 1)) for h in H)
    res["seeded"]["query_support"] = [sup_seed, len(H)]
    sup_merge = sum(q in model.blocks(t) for t in T)
    res["merger"]["query_support"] = [sup_merge, len(T)]
    # seed-anchored trees among merger trees (Proposition 1)
    cater = sum(all((model.seed in A and len(B) == 1) or (model.seed in B and len(A) == 1) for A, B, _ in t)
                for t in T)
    res["merger"]["seed_caterpillars"] = cater
    # root splits
    roots = {}
    for t in T:
        A, B, _ = t[-1]
        key = " | ".join(sorted(["{" + ",".join(sorted(A)) + "}", "{" + ",".join(sorted(B)) + "}"]))
        roots[key] = roots.get(key, 0) + 1
    res["merger"]["root_splits"] = roots
    # naive vs flow
    res["seeded"]["naive_vs_flow"] = [
        {"history": list(h), "naive": str(model.naive_step_probability(h)),
         "flow": str(model.flow_step_probability(h))} for h in H]
    # perturbations
    res["perturbations"] = []
    for e in model.contacts:
        m2 = model.without(e)
        res["perturbations"].append({"removed": list(e), "seeded": len(m2.seeded_histories()),
                                     "merger": len(m2.merger_trees())})
    # crossings per merge
    patterns = {}
    pairs_closed = set()
    for t in T:
        key = tuple(sorted(len(c) for *_, c in t))
        patterns[str(list(key))] = patterns.get(str(list(key)), 0) + 1
        for *_, c in t:
            if len(c) > 1:
                pairs_closed.add(tuple("-".join(e) for e in c))
    res["merger"]["crossing_count_patterns"] = patterns
    res["merger"]["multi_contact_sets"] = sorted([list(p) for p in pairs_closed])
    return res


# =============================================================================
# 5. Page rendering (plain A4 pages; matplotlib only)
# =============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

INK, MUTED, ACC, SEEDC, FAINT = "#1f2328", "#6b7280", "#0f766e", "#2563eb", "#e5e7eb"
A4 = (8.27, 11.69)


class Page:
    """A very small top-to-bottom text layout engine."""

    def __init__(self, pdf, title, subtitle, footer):
        self.pdf, self.title, self.footer = pdf, title, footer
        self._new(title, subtitle)

    def _new(self, title, subtitle):
        self.fig = plt.figure(figsize=A4)
        self.ax = self.fig.add_axes([0, 0, 1, 1]); self.ax.set_axis_off()
        self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1)
        self.y = 0.95
        self.ax.text(0.08, self.y, title, fontsize=16, fontweight="bold", va="top", color=INK)
        self.y -= 0.035
        self.ax.text(0.08, self.y, subtitle, fontsize=9.5, va="top", color=MUTED)
        self.y -= 0.035
        self.ax.plot([0.08, 0.92], [self.y, self.y], color=FAINT, lw=1)
        self.y -= 0.02
        self.ax.text(0.08, 0.03, self.footer, fontsize=7, color=MUTED)

    def ensure(self, h):
        """Start a continuation page if fewer than h of the page remains."""
        if self.y - h < 0.07:
            self.pdf.savefig(self.fig); plt.close(self.fig)
            self._new(self.title + " (continued)", "")

    def heading(self, s, col=ACC):
        self.ensure(0.08)
        self.y -= 0.01
        self.ax.text(0.08, self.y, s, fontsize=11.5, fontweight="bold", va="top", color=col)
        self.y -= 0.028

    def para(self, s, size=9.5, col=INK, width=96, indent=0.0, gap=0.008):
        for line in textwrap.wrap(s, width=width):
            self.ensure(0.02)
            self.ax.text(0.08 + indent, self.y, line, fontsize=size, va="top", color=col)
            self.y -= 0.0185 * size / 9.5
        self.y -= gap

    def bullets(self, items, size=9.5, width=90):
        for it in items:
            lines = textwrap.wrap(it, width=width)
            self.ensure(0.02 * len(lines))
            self.ax.text(0.09, self.y, "•", fontsize=size, va="top", color=INK)
            for line in lines:
                self.ax.text(0.11, self.y, line, fontsize=size, va="top", color=INK)
                self.y -= 0.0185 * size / 9.5
            self.y -= 0.004
        self.y -= 0.006

    def lines(self, n, h=0.03):
        for _ in range(n):
            if self.y - h < 0.07:      # writing lines are optional; never start a page for them
                break
            self.y -= h
            self.ax.plot([0.1, 0.92], [self.y, self.y], color=FAINT, lw=0.8)
        self.y -= 0.012

    def table(self, header, rows, widths, size=8.8):
        self.ensure(0.022 + 0.02 * len(rows))
        x0 = 0.1
        xs = [x0]
        for w in widths[:-1]:
            xs.append(xs[-1] + w)
        for x, h in zip(xs, header):
            self.ax.text(x, self.y, h, fontsize=size, fontweight="bold", va="top")
        self.y -= 0.022
        self.ax.plot([x0, x0 + sum(widths)], [self.y + 0.004, self.y + 0.004], color=FAINT, lw=0.8)
        for r in rows:
            for x, c in zip(xs, r):
                self.ax.text(x, self.y, str(c), fontsize=size, va="top")
            self.y -= 0.02
        self.y -= 0.01

    def graph(self, model, height=0.16, highlight=()):
        """Draw the contact graph (a contact map, not an order). Acyclic maps
        are drawn as a tree hanging from the seed; others on a circle."""
        self.ensure(height + 0.03)
        n = len(model.components)
        cx, cy, r = 0.5, self.y - height / 2, height * 0.38
        P = {}
        if model.n_cycles() == 0:
            depth, order, frontier = {model.seed: 0}, [model.seed], [model.seed]
            while frontier:
                nxt = []
                for u in frontier:
                    for a, b in model.contacts:
                        for x, y in ((a, b), (b, a)):
                            if x == u and y not in depth:
                                depth[y] = depth[u] + 1; nxt.append(y); order.append(y)
                frontier = nxt
            # x position: leaves spread left to right in DFS order
            kids = {u: [v for v in order if depth[v] == depth[u] + 1 and tuple(sorted((u, v))) in model.contacts]
                    for u in order}
            leaves = []

            def dfs(u):
                if not kids[u]:
                    leaves.append(u)
                for v in kids[u]:
                    dfs(v)
            dfs(model.seed)
            xs = {}

            def place(u):
                if not kids[u]:
                    xs[u] = leaves.index(u)
                else:
                    for v in kids[u]:
                        place(v)
                    xs[u] = sum(xs[v] for v in kids[u]) / len(kids[u])
            place(model.seed)
            span = max(1, len(leaves) - 1); D = max(depth.values()) or 1
            for c in model.components:
                P[c] = (0.5 + 0.34 * (xs[c] / span - 0.5) if len(leaves) > 1 else 0.5,
                        self.y - 0.015 - (height - 0.04) * depth[c] / D)
        else:
            for i, c in enumerate(model.components):
                ang = math.pi / 2 - 2 * math.pi * i / n
                P[c] = (cx + r * math.cos(ang) * 1.414, cy + r * math.sin(ang))
        for u, v in model.contacts:
            hot = tuple(sorted((u, v))) in [tuple(sorted(h)) for h in highlight]
            self.ax.plot(*zip(P[u], P[v]), color="#d97706" if hot else "#9ca3af", lw=2.4 if hot else 1.4, zorder=1)
        for c, (x, y) in P.items():
            seed = c == model.seed
            self.ax.text(x, y, c, ha="center", va="center", fontsize=9, zorder=3,
                         color=SEEDC if seed else INK, fontweight="bold" if seed else "normal",
                         bbox=dict(boxstyle="round,pad=0.3", fc="#dbeafe" if seed else "white",
                                   ec=SEEDC if seed else INK, lw=0.9))
        self.ax.text(cx, self.y - height + 0.002, "contact map (lines = declared contacts; not an assembly order)",
                     ha="center", fontsize=7.5, color=MUTED)
        self.y -= height + 0.02

    def close(self):
        self.pdf.savefig(self.fig); plt.close(self.fig)


def _fmt(h):
    return " → ".join(h)


def _pair(q):
    return "–".join(q)


FOOT = "Protein Complex Jigsaw teaching pack · every number on this page is computed by exact enumeration (pcjigsaw.education)"


# -----------------------------------------------------------------------------
# 5a. Outreach card (school level)
# -----------------------------------------------------------------------------
def outreach(model, R, out):
    seed, q = model.seed, _pair(model.query)
    n_seed, n_merge = R["seeded"]["count"], R["merger"]["count"]
    s_q, _ = R["merger"]["query_support"]
    with PdfPages(out) as pdf:
        p = Page(pdf, "Build a protein machine", f"A jigsaw activity · model: {model.name}", FOOT)
        p.para("Many proteins only work when several of them join together into a 'complex'. "
               "Scientists want to know the ORDER in which the pieces join. In this activity you "
               "are the cell: you build the complex from cut-out pieces and find out how many "
               "different ways there are to build it.")
        p.graph(model, height=0.13)
        p.heading("Round 1 · Grow from one piece")
        p.bullets([f"Start with the {seed} piece (the blue one on the map).",
                   "Add ONE piece at a time. A new piece must click onto a piece that is already there.",
                   "Write down your order, then take it apart and try to find a DIFFERENT order."])
        p.para("My orders:", size=9.5, gap=0)
        p.lines(3, h=0.024)
        p.heading("Round 2 · Class count")
        p.para("Put every different order the class found on the board. Do you think you have found them all? "
               "How could you be sure?")
        p.heading("Round 3 · Change one rule")
        p.bullets(["New rule: you may now build SMALL groups of pieces separately and click the groups together.",
                   f"Can you build the {q} pair first, on its own? Was that possible in Round 1?",
                   "Scientists have detected this pair in experiments. Which rule allows it?"])
        p.lines(2, h=0.026)
        p.heading("Round 4 · Break a connection")
        p.para("Imagine one connection on the map is broken by a mutation. Choose one. Can the complex still be built?")
        p.lines(1, h=0.026)
        p.close()

        f = Page(pdf, "Facilitator notes", f"Answers for: {model.name}", FOOT)
        f.heading("Round 1 and 2 · the answer")
        f.para(f"With the 'grow from {seed}' rule there are exactly {n_seed} different orders:")
        for h in R["seeded"]["histories"][:12]:
            f.para("    " + _fmt(h), size=9)
        if n_seed > 12:
            f.para(f"    … ({n_seed} in total)", size=9)
        f.para(f"Without the 'must click on' rule there would be {R['all_orders_from_seed']} orders "
               f"after {seed}; most of them are impossible because a piece would float in the air.")
        f.heading("Round 3 · the answer")
        f.para(f"With the group rule there are {n_merge} different ways to build it, and the {q} pair is built "
               f"as a separate group in {s_q} of them. In Round 1 it can never form on its own, because every step "
               f"must include {seed}. Changing the RULE (the 'representation'), not the pieces, changes what is possible "
               "– this is the main idea of the research paper.")
        f.heading("Round 4 · the answer")
        rows = [(_pair(d["removed"]), d["seeded"], d["merger"]) for d in R["perturbations"]]
        f.table(["connection broken", f"orders from {seed}", "group-rule ways"], rows, [0.3, 0.2, 0.2])
        f.heading("Talking points (about 5 minutes)")
        f.bullets(["Counting carefully beats guessing: a computer lists every possibility, so we know when we have all of them.",
                   "A model is a set of rules we choose to believe. Different rules give different answers; scientists check which rules match experiments.",
                   "The number of ways is not how often each way happens in a cell – it tells us what is possible."])
        f.close()


# -----------------------------------------------------------------------------
# 5b. Undergraduate worksheet + answer key
# -----------------------------------------------------------------------------
def undergraduate(model, R, ring=None, Rring=None, chain=None, Rchain=None, out_ws=None, out_ans=None):
    seed, q = model.seed, _pair(model.query)
    n = len(model.components)

    def exercises(p, answers):
        p.para("Learning goals: (i) define an assembly space from declared contacts; (ii) see how the choice of "
               "representation changes which intermediates are possible; (iii) compute supports and perturbation "
               "effects exactly; (iv) recognise when a scoring rule cannot distinguish assembly histories.", size=9, col=MUTED)
        p.graph(model, highlight=[model.query])
        p.heading("1 · Seed-anchored histories")
        p.para(f"Start from {seed}. At each step add one component that contacts the current complex. "
               f"List every possible order. How many are there, and what fraction of the {R['all_orders_from_seed']} "
               f"orderings of the other {n - 1} components is this?")
        if answers:
            p.para(f"Answer: {R['seeded']['count']} histories, {R['seeded']['count']}/{R['all_orders_from_seed']} of all orderings.",
                   col=ACC)
            for h in R["seeded"]["histories"][:8]:
                p.para("    " + _fmt(h), size=8.5, col=ACC, gap=0)
            p.y -= 0.01
        else:
            p.lines(3)
        p.heading("2 · Merger trees and an off-seed intermediate")
        p.para(f"Now allow two connected blocks to merge whenever they share a contact. Count the binary merger trees "
               f"(count each tree once). In how many does {q} form as an intermediate? Why can {q} never form in exercise 1?")
        if answers:
            sq, nq = R["merger"]["query_support"]
            p.para(f"Answer: {R['merger']['count']} merger trees; {q} occurs in {sq}/{nq}. In exercise 1 every "
                   f"intermediate contains {seed}, so its support is {R['seeded']['query_support'][0]}/{R['seeded']['count']}. "
                   f"The {R['merger']['seed_caterpillars']} merger trees that add one component at a time to the {seed} "
                   "branch are exactly the seed-anchored histories (Proposition 1).", col=ACC)
            deg = {c: sum(c in e for e in model.contacts) for c in model.components}
            if R["independent_cycles"] == 0 and max(deg.values()) <= 2:
                p.para(f"Hint for checking: the contact graph is a tree; for a chain of {n} components the count is the "
                       f"Catalan number C({n - 1}) = {math.comb(2 * (n - 1), n - 1) // n}.", size=8.5, col=ACC)
        else:
            p.lines(3)
        p.heading("3 · Perturbation")
        p.para("Remove each contact in turn and recount both spaces. Which contacts are essential? "
               "What does your answer say about complexes whose contact map has no cycles?")
        if answers:
            rows = [(_pair(d["removed"]), d["seeded"], d["merger"]) for d in R["perturbations"]]
            p.table(["contact removed", "seed-anchored", "merger"], rows, [0.3, 0.18, 0.18])
            if R["independent_cycles"] == 0:
                p.para("Every contact is essential: in an acyclic map, removing any contact disconnects the complex.",
                       col=ACC)
        else:
            p.lines(3)

    def exercises2(p, answers):
        p.heading("4 · Building at random is not sampling at random")
        p.para(f"A student builds seed-anchored histories by choosing uniformly among the legal next components at "
               f"every step. What is the probability of each history? Is it 1/{R['seeded']['count']}? "
               "How must the choice be weighted so that every history is equally likely?")
        if answers:
            rows = [(_fmt(d["history"][1:]), d["naive"], d["flow"]) for d in R["seeded"]["naive_vs_flow"][:10]]
            p.table(["history (after the seed)", "uniform step", "exact flow"], rows, [0.48, 0.14, 0.14], size=8.2)
            p.para("Weighting each step by the number of completions below it (the exact flow F(child)/F(state)) "
                   "makes every history equally likely (Proposition 2).", col=ACC)
        else:
            p.lines(4)
        if ring is not None:
            p.heading("5 · When can a score tell trees apart?")
            p.para(f"Compare the chain {'–'.join(chain.components)} with the ring {'–'.join(ring.components)}–{ring.components[0]}. "
                   "For every merger tree, write down how many contacts each merge closes. Could a score that adds a fixed "
                   "amount per contact ever prefer one tree? Could a score that depends on WHICH contacts are closed together?")
            if answers:
                p.para(f"Answer: chain — {Rchain['merger']['count']} trees, crossing pattern(s) "
                       f"{', '.join(Rchain['merger']['crossing_count_patterns'])} (every merge closes one contact). "
                       f"Ring — {Rring['merger']['count']} trees, pattern(s) {', '.join(Rring['merger']['crossing_count_patterns'])}: "
                       "each tree has one merge closing two contacts, so a count-only score is still constant, but the pair "
                       f"closed differs ({len(Rring['merger']['multi_contact_sets'])} different pairs). A fixed per-contact score "
                       "is constant on any graph (Proposition 4); contact-specific interface scores need cycles, and cycles "
                       "are not sufficient (Proposition 5).", col=ACC)
            else:
                p.lines(4)

    for out, answers in ((out_ws, False), (out_ans, True)):
        with PdfPages(out) as pdf:
            t = "Assembly spaces: worksheet" if not answers else "Assembly spaces: answer key"
            p = Page(pdf, t, f"Undergraduate exercises · model: {model.name}", FOOT)
            exercises(p, answers)
            exercises2(p, answers)
            p.close()


# =============================================================================
# 6. Driver
# =============================================================================
def build_pack(model_dir, out_dir, seed=None, query=None, kits=True):
    os.makedirs(out_dir, exist_ok=True)
    model_dir = model_path(model_dir)
    model = TeachingModel.from_dir(model_dir, seed=seed, query=query)
    R = compute(model)
    toys = os.path.join(DATA, "models")
    chain = TeachingModel.from_dir(os.path.join(toys, "chain4"))
    ring = TeachingModel.from_dir(os.path.join(toys, "cycle4"))
    Rc, Rr = compute(chain), compute(ring)
    with open(os.path.join(out_dir, "answers.json"), "w") as fh:
        json.dump({"model": R, "chain4": Rc, "cycle4": Rr}, fh, indent=1)
    outreach(model, R, os.path.join(out_dir, "outreach_card.pdf"))
    undergraduate(model, R, ring, Rr, chain, Rc,
                  os.path.join(out_dir, "worksheet.pdf"), os.path.join(out_dir, "answers.pdf"))
    if kits:
        from scijigsaw.tiles import tiles
        tiles(os.path.join(model_dir, "proteins.csv"), os.path.join(model_dir, "interactions.csv"),
              os.path.join(out_dir, "kit.pdf"), title=model.name)
    return R
