"""Exact generative flow over admissible protein-complex assembly mechanisms.

PCJ Flow separates two operations:

1. the declared biological model determines which construction steps are admissible;
2. a non-negative local reward redistributes flow only inside that admissible space.

No neural network is required for the exact engines in this module.  The seed
engine uses a continuation dynamic programme over partial seed-containing
assemblies.  The merger engine uses the exact binary-tree recurrence and samples
root splits in proportion to their recursively accumulated flow.

The public API intentionally consumes the existing ``scijigsaw.assembly.Assembly``
object.  This keeps all current biological encodings and published regression
counts valid while moving the new computational surface into ``pcjigsaw``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import random
from typing import Callable, FrozenSet, Hashable, Iterable, Mapping, Protocol, Sequence

Component = str
Subset = FrozenSet[Component]
SeedWeight = Callable[[Subset, Component], float]
MergeWeight = Callable[[Subset, Subset, Subset], float]


def _unit_weight(*_args) -> float:
    return 1.0


def _finite_nonnegative(value: float, label: str = "weight") -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and non-negative, got {value!r}")
    return value


def _excluded(model, left: Iterable[str], right: Iterable[str]) -> bool:
    left, right = set(left), set(right)
    for a, b in getattr(model, "excludes", ()):
        if (left & set(a) and right & set(b)) or (left & set(b) and right & set(a)):
            return True
    return False


@dataclass(frozen=True)
class SeedStep:
    state: Subset
    component: Component
    next_state: Subset
    weight: float
    continuation_flow: float
    probability: float


@dataclass(frozen=True)
class MergerSplit:
    parent: Subset
    left: Subset
    right: Subset
    weight: float
    left_flow: float
    right_flow: float
    probability: float


@dataclass(frozen=True)
class FlowSummary:
    representation: str
    root_flow: float
    exact: bool
    uniform: bool
    states_evaluated: int


@dataclass(frozen=True)
class ModelFlow:
    """Flow result for one explicitly declared hard model.

    Model alternatives are intentionally represented by separate ``ModelFlow``
    values.  PCJ Flow never turns uncertainty about a hard relation into a weak
    edge inside one graph.
    """

    label: str
    summary: FlowSummary


class ExactSeedFlow:
    """Exact flow over seed-anchored assembly histories.

    The seed is implicit and is not part of ``state``.  Under unit transition
    weight, ``flow(S)`` is exactly the number of admissible complete histories
    continuing from ``S``.  Therefore ``flow(frozenset())`` reproduces
    ``Assembly.n_orders_permitted(contact_connected=True)``, which equals the
    legacy count for ``requires`` encodings.

    A non-uniform ``step_weight(S, u)`` defines the product reward of a terminal
    history.  It can redistribute mass among admissible histories but can never
    make a forbidden addition legal.
    """

    def __init__(self, model, step_weight: SeedWeight | None = None):
        self.model = model
        self.seed = getattr(model, "seed", None)
        # The seed is present from the start and is never added, even when a
        # typed encoding lists it among the units (it appears in a contact).
        self.units = tuple(u for u in getattr(model, "units", ()) if u != self.seed)
        self.full = frozenset(self.units)
        self.step_weight = step_weight or _unit_weight
        self._contacts = {frozenset(e) for e in getattr(model, '_contacts', ()) if len(e) == 2}
        self._visited: set[Subset] = set()

    def _has_contact(self, a: str, b: str) -> bool:
        return frozenset((a, b)) in self._contacts

    def _contact_connected_addition(self, state: Subset, component: Component) -> bool:
        """Require a seeded addition to contact the assembled complex.

        This is the software condition used in Proposition 1: a seed-anchored
        history is exactly the restricted merger family in which every merger
        joins one singleton to the seed-containing branch.  Legacy
        ``Assembly.n_orders_permitted`` remains unchanged for backwards
        compatibility; PCJ Flow uses this stricter contact-connected semantics.
        """
        if self.seed is None:
            return True
        assembled = set(state) | {self.seed}
        return any(self._has_contact(component, v) for v in assembled)

    def admissible_additions(self, state: Subset) -> tuple[Component, ...]:
        state = frozenset(state)
        out = []
        for u in self.units:
            if u in state:
                continue
            req = set(getattr(self.model, "requires", {}).get(u, set()))
            if self.seed is not None:
                req.discard(self.seed)
            # Ignore non-unit sentinel values exactly as the historical forward
            # state graph does; a sentinel remains missing and therefore blocks.
            missing = {r for r in req if r not in state}
            if missing:
                continue
            if _excluded(self.model, {u}, state):
                continue
            if not self._contact_connected_addition(state, u):
                continue
            out.append(u)
        return tuple(sorted(out))

    @lru_cache(maxsize=None)
    def flow(self, state: Subset = frozenset()) -> float:
        state = frozenset(state)
        self._visited.add(state)
        if state == self.full:
            return 1.0
        total = 0.0
        for u in self.admissible_additions(state):
            nxt = state | {u}
            w = _finite_nonnegative(self.step_weight(state, u), "step weight")
            if w:
                total += w * self.flow(nxt)
        return total

    def transitions(self, state: Subset = frozenset()) -> tuple[SeedStep, ...]:
        state = frozenset(state)
        z = self.flow(state)
        if z <= 0:
            return ()
        rows = []
        for u in self.admissible_additions(state):
            nxt = state | {u}
            w = _finite_nonnegative(self.step_weight(state, u), "step weight")
            q = self.flow(nxt)
            mass = w * q
            if mass > 0:
                rows.append(SeedStep(state, u, nxt, w, q, mass / z))
        return tuple(rows)

    def sample_history(self, rng: random.Random | None = None) -> tuple[Component, ...]:
        rng = rng or random.Random()
        state: Subset = frozenset()
        history: list[Component] = []
        if self.flow(state) <= 0:
            raise ValueError("no admissible seed-anchored completion")
        while state != self.full:
            rows = self.transitions(state)
            if not rows:
                raise RuntimeError("positive flow state has no positive-flow transition")
            x = rng.random()
            acc = 0.0
            chosen = rows[-1]
            for row in rows:
                acc += row.probability
                if x <= acc:
                    chosen = row
                    break
            history.append(chosen.component)
            state = chosen.next_state
        return tuple(history)

    def summary(self) -> FlowSummary:
        z = self.flow(frozenset())
        uniform = self.step_weight is _unit_weight
        return FlowSummary("seed", z, True, uniform, len(self._visited))


class ExactMergerFlow:
    """Exact generative flow over admissible binary merger trees.

    ``flow(S)`` is the total reward of all admissible binary trees whose leaves
    are exactly ``S``.  Under unit merge weight it is the exact tree count and
    reproduces ``Assembly.n_trees()``.

    The recursive root-split construction gives each unordered binary merger tree
    one representation; it therefore avoids the trajectory-multiplicity bias that
    would arise from a naive random walk over arbitrary forest merge orderings.
    """

    def __init__(self, model, merge_weight: MergeWeight | None = None):
        self.model = model
        units = list(getattr(model, "units", ()))
        seed = getattr(model, "seed", None)
        if seed is not None and seed not in units:
            units.append(seed)
        self.units = tuple(units)
        self.full = frozenset(self.units)
        self.merge_weight = merge_weight or _unit_weight
        self._contacts = {frozenset(e) for e in getattr(model, "_contacts", ()) if len(e) == 2}
        self._visited: set[Subset] = set()

    def _has_contact(self, a: str, b: str) -> bool:
        return frozenset((a, b)) in self._contacts

    def connected(self, subset: Subset) -> bool:
        subset = frozenset(subset)
        if not subset:
            return False
        seen = {next(iter(subset))}
        stack = list(seen)
        while stack:
            u = stack.pop()
            for v in subset - seen:
                if self._has_contact(u, v):
                    seen.add(v)
                    stack.append(v)
        return seen == set(subset)

    def valid_intermediate(self, subset: Subset) -> bool:
        subset = frozenset(subset)
        if len(subset) == 1:
            return True
        if not self.connected(subset):
            return False
        if _excluded(self.model, subset, subset):
            # _excluded on the same set would also match the same side twice;
            # perform the intended internal exclusion test explicitly below.
            pass
        for a, b in getattr(self.model, "excludes", ()):
            if subset & set(a) and subset & set(b):
                return False
        requires = getattr(self.model, "requires", {})
        for q in subset:
            req = set(requires.get(q, set()))
            if not req.issubset(subset):
                return False
        return True

    def admissible_splits(self, parent: Subset) -> tuple[tuple[Subset, Subset], ...]:
        parent = frozenset(parent)
        if len(parent) <= 1 or not self.valid_intermediate(parent):
            return ()
        anchor = min(parent)
        rest = sorted(parent - {anchor})
        out: list[tuple[Subset, Subset]] = []
        # Enumerate subsets containing anchor. This canonicalises A|B vs B|A.
        for mask in range(1 << len(rest)):
            left = {anchor}
            for i, u in enumerate(rest):
                if mask >> i & 1:
                    left.add(u)
            A = frozenset(left)
            B = parent - A
            if not B:
                continue
            if not self.valid_intermediate(A) or not self.valid_intermediate(B):
                continue
            if not any(self._has_contact(u, v) for u in A for v in B):
                continue
            out.append((A, B))
        return tuple(out)

    @lru_cache(maxsize=None)
    def flow(self, subset: Subset | None = None) -> float:
        subset = self.full if subset is None else frozenset(subset)
        self._visited.add(subset)
        if len(subset) == 1:
            return 1.0
        if not self.valid_intermediate(subset):
            return 0.0
        total = 0.0
        for A, B in self.admissible_splits(subset):
            w = _finite_nonnegative(self.merge_weight(A, B, subset), "merge weight")
            if w:
                total += w * self.flow(A) * self.flow(B)
        return total

    def splits(self, parent: Subset | None = None) -> tuple[MergerSplit, ...]:
        parent = self.full if parent is None else frozenset(parent)
        z = self.flow(parent)
        if z <= 0:
            return ()
        rows = []
        for A, B in self.admissible_splits(parent):
            w = _finite_nonnegative(self.merge_weight(A, B, parent), "merge weight")
            fa, fb = self.flow(A), self.flow(B)
            mass = w * fa * fb
            if mass > 0:
                rows.append(MergerSplit(parent, A, B, w, fa, fb, mass / z))
        return tuple(rows)

    def sample_tree(self, rng: random.Random | None = None, subset: Subset | None = None):
        """Sample an admissible nested binary tree exactly in proportion to reward."""
        rng = rng or random.Random()
        subset = self.full if subset is None else frozenset(subset)
        if len(subset) == 1:
            return next(iter(subset))
        rows = self.splits(subset)
        if not rows:
            raise ValueError(f"no admissible merger tree for {sorted(subset)}")
        x = rng.random()
        acc = 0.0
        chosen = rows[-1]
        for row in rows:
            acc += row.probability
            if x <= acc:
                chosen = row
                break
        return (
            self.sample_tree(rng, chosen.left),
            self.sample_tree(rng, chosen.right),
        )

    @lru_cache(maxsize=None)
    def _containing_flow(self, event: Subset, parent: Subset) -> float:
        event, parent = frozenset(event), frozenset(parent)
        if not event.issubset(parent):
            return 0.0
        if parent == event:
            return self.flow(parent)
        if len(parent) == 1 or not self.valid_intermediate(parent):
            return 0.0
        total = 0.0
        for A, B in self.admissible_splits(parent):
            w = _finite_nonnegative(self.merge_weight(A, B, parent), "merge weight")
            if event.issubset(A):
                total += w * self._containing_flow(event, A) * self.flow(B)
            elif event.issubset(B):
                total += w * self.flow(A) * self._containing_flow(event, B)
        return total

    def support(self, event: Iterable[str]) -> dict:
        """Reward-normalised support for a subcomplex event.

        With unit reward this exactly reduces to the historical count ratio.
        """
        event = frozenset(event)
        z = self.flow(self.full)
        if z <= 0:
            return {"subset": sorted(event), "flow_containing": 0.0,
                    "flow_total": 0.0, "support": float("nan"),
                    "reading": "no admissible tree"}
        if not event.issubset(self.full) or (len(event) > 1 and not self.connected(event)):
            h = 0.0
        else:
            h = self._containing_flow(event, self.full)
        p = h / z
        eps = 1e-12
        reading = "impossible" if h <= eps else "necessary" if abs(h-z) <= eps*max(1,z) else "optional"
        return {"subset": sorted(event), "flow_containing": h,
                "flow_total": z, "support": p, "reading": reading}

    def edge_additive_degeneracy(self, edge_weights: Mapping[FrozenSet[str], float] | None = None) -> dict:
        """Diagnose why fixed pairwise edge-additive rewards cannot rank full trees.

        In every complete binary merger tree, the endpoints of each final contact
        first become members of the same branch at exactly one internal merge.
        Therefore a reward that sums a fixed weight for each newly realised final
        contact has the same terminal total for every complete tree.
        """
        weights = edge_weights or {e: 1.0 for e in self._contacts}
        score = 0.0
        for e in self._contacts:
            score += _finite_nonnegative(weights.get(e, 0.0), "edge weight")
        return {
            "non_identifying": True,
            "constant_terminal_score": score,
            "n_contacts": len(self._contacts),
            "reason": ("fixed pairwise contact contributions are each realised exactly once "
                       "in every complete merger tree; use merge-, state-, or terminal-dependent features"),
        }

    def summary(self) -> FlowSummary:
        z = self.flow(self.full)
        uniform = self.merge_weight is _unit_weight
        return FlowSummary("merger", z, True, uniform, len(self._visited))



def plackett_luce_step_weight(abundance: Mapping[str, float], units: Sequence[str]) -> SeedWeight:
    """Return the history-dependent Plackett--Luce encounter factor.

    The denominator contains all components not yet incorporated, matching the
    standard Plackett--Luce probability of a complete encounter order before
    conditioning on biological admissibility.  Multiplying all abundances by a
    common positive constant leaves every factor, and therefore the complete
    PCJ Flow distribution, unchanged.
    """
    vals = {u: _finite_nonnegative(abundance[u], f"abundance[{u}]") for u in units}
    if any(v <= 0 for v in vals.values()):
        raise ValueError("Plackett--Luce abundances must be strictly positive")
    units = tuple(units)
    def weight(state: Subset, u: str) -> float:
        remaining = [v for v in units if v not in state]
        denom = sum(vals[v] for v in remaining)
        return vals[u] / denom
    return weight


def interface_potential(edge_recurrence: Mapping[FrozenSet[str], float],
                        edge_residues: Mapping[FrozenSet[str], float],
                        lam: float = 1.0) -> MergeWeight:
    """Merge-dependent interface potential used for reward-identifiability tests.

    For a merge A+B, crossing contacts receive score
        s_m = mean(recurrence_e) + log(1 + median(residues_e)),
    and potential exp(lam * s_m).  On an acyclic contact graph every admissible
    merge is crossed by exactly one contact, so the product is constant across
    complete trees; cycles can expose multi-contact interfaces and make the
    feature history-dependent.
    """
    lam = float(lam)
    def weight(A: Subset, B: Subset, parent: Subset) -> float:
        crossing = [e for e in edge_recurrence if e & A and e & B]
        if not crossing:
            return 0.0
        rec = [float(edge_recurrence[e]) for e in crossing]
        res = sorted(float(edge_residues.get(e, 0.0)) for e in crossing)
        mid = len(res)//2
        median = res[mid] if len(res)%2 else (res[mid-1]+res[mid])/2
        score = sum(rec)/len(rec) + math.log1p(median)
        return math.exp(lam * score)
    return weight


def effective_number(probabilities: Iterable[float]) -> float:
    """Entropy-equivalent number of mechanisms exp(H)."""
    ps = [float(p) for p in probabilities if p > 0]
    z = sum(ps)
    if z <= 0:
        return 0.0
    ps = [p/z for p in ps]
    return math.exp(-sum(p * math.log(p) for p in ps))


def compare_model_flows(models: Mapping[str, object], representation: str = "seed") -> tuple[ModelFlow, ...]:
    """Evaluate alternative hard models separately; no implicit model averaging."""
    out = []
    for label, model in models.items():
        if representation == "seed":
            engine = ExactSeedFlow(model)
        elif representation == "merger":
            engine = ExactMergerFlow(model)
        else:
            raise ValueError("representation must be 'seed' or 'merger'")
        out.append(ModelFlow(label, engine.summary()))
    return tuple(out)


__all__ = [
    "ExactSeedFlow", "ExactMergerFlow", "FlowSummary", "SeedStep",
    "MergerSplit", "ModelFlow", "compare_model_flows",
    "plackett_luce_step_weight", "interface_potential", "effective_number",
]
