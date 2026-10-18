"""Variant-to-mechanism perturbation analysis for Protein Complex Jigsaw.

This module deliberately does not infer pathogenicity from sequence. It accepts
an explicitly declared molecular perturbation (lost component, lost contact, or
added exclusion), recomputes the assembly model, and reports mechanistic changes.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Iterable, Mapping

from scijigsaw.assembly import Assembly


@dataclass(frozen=True)
class Perturbation:
    """A declared molecular perturbation.

    Parameters are intentionally mechanism-level rather than sequence-level.
    Variant-effect predictors or experiments should be used upstream to justify
    these declarations.
    """
    label: str
    unavailable_components: tuple[str, ...] = ()
    lost_contacts: tuple[tuple[str, str], ...] = ()
    added_exclusions: tuple[tuple[str, str], ...] = ()
    provenance: str | None = None


@dataclass
class VariantImpactReport:
    label: str
    wt_orders: int
    perturbed_orders: int
    wt_trees: int | None
    perturbed_trees: int | None
    full_target_reachable: bool
    largest_reachable_state: tuple[str, ...]
    trapped_intermediates: tuple[tuple[str, ...], ...]
    first_blocked_transitions: tuple[tuple[tuple[str, ...], str, str], ...]
    unavailable_interfaces: tuple[tuple[str, str], ...]
    surviving_productive_route: bool
    provenance: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _canonical_edge(a: str, b: str) -> frozenset[str]:
    return frozenset((a, b))


def _clone_with_perturbation(model: Assembly, p: Perturbation) -> Assembly:
    unavailable = set(p.unavailable_components)
    lost = {_canonical_edge(a, b) for a, b in p.lost_contacts}

    req = {}
    for u, ps in model.requires.items():
        if u in unavailable:
            continue
        kept = {x for x in ps if x not in unavailable and _canonical_edge(u, x) not in lost}
        req[u] = kept

    contacts = []
    for fs in model._contacts:
        vals = tuple(fs)
        if len(vals) != 2:
            continue
        a, b = vals
        if a in unavailable or b in unavailable or fs in lost:
            continue
        contacts.append((a, b))

    exclusions = []
    for a, b in model.excludes:
        aa, bb = set(a) - unavailable, set(b) - unavailable
        if aa and bb:
            exclusions.append((aa, bb))
    exclusions.extend(({a}, {b}) for a, b in p.added_exclusions
                      if a not in unavailable and b not in unavailable)

    observed = [set(x) - unavailable for x in getattr(model, "observed", [])]
    units = [u for u in model.units if u not in unavailable]

    return Assembly(
        requires=req,
        excludes=exclusions,
        seed=model.seed if model.seed not in unavailable else None,
        contacts=contacts,
        units=units,
        provenance=getattr(model, "provenance", {}),
        observed_subcomplexes=observed,
    )


def _reachable_states(model: Assembly) -> set[frozenset[str]]:
    """Enumerate states reachable by legal singleton additions.

    This is the seed-anchored state graph corresponding to the declared
    prerequisite/exclusion model; it is separate from merger-tree enumeration.
    """
    units = tuple(model.units)
    seen = {frozenset()}
    frontier = [frozenset()]
    while frontier:
        s = frontier.pop()
        for u in units:
            if u in s:
                continue
            req = {x for x in model.requires.get(u, set()) if x in units}
            if not req.issubset(s):
                continue
            blocked = False
            for a, b in model.excludes:
                if u in a and (set(b) & set(s)):
                    blocked = True
                if u in b and (set(a) & set(s)):
                    blocked = True
            if blocked:
                continue
            ns = frozenset(set(s) | {u})
            if ns not in seen:
                seen.add(ns)
                frontier.append(ns)
    return seen


def _block_reason(model: Assembly, state: frozenset[str], u: str) -> str:
    missing = sorted(x for x in model.requires.get(u, set()) if x in model.units and x not in state)
    if missing:
        return "missing prerequisite(s): " + ", ".join(missing)
    for a, b in model.excludes:
        if u in a and (set(b) & set(state)):
            return "excluded by: " + ", ".join(sorted(set(b) & set(state)))
        if u in b and (set(a) & set(state)):
            return "excluded by: " + ", ".join(sorted(set(a) & set(state)))
    return "not blocked"


def analyse_perturbation(model: Assembly, perturbation: Perturbation) -> VariantImpactReport:
    """Compare a declared perturbation with the wild-type mechanism space."""
    pert = _clone_with_perturbation(model, perturbation)
    wt_orders = model.n_orders_permitted()
    pert_orders = pert.n_orders_permitted()

    try:
        wt_trees = model.n_trees()
    except Exception:
        wt_trees = None
    try:
        pert_trees = pert.n_trees()
    except Exception:
        pert_trees = None

    states = _reachable_states(pert)
    target = frozenset(pert.units)
    full = target in states
    max_n = max((len(s) for s in states), default=0)
    maximal = sorted((tuple(sorted(s)) for s in states if len(s) == max_n))

    blocked = []
    for s in states:
        if len(s) != max_n:
            continue
        for u in pert.units:
            if u not in s:
                reason = _block_reason(pert, s, u)
                if reason != "not blocked":
                    blocked.append((tuple(sorted(s)), u, reason))

    # Lost interfaces are reported from the declaration, plus all interfaces
    # touching a component declared unavailable.
    unavailable_ifaces = {_canonical_edge(a, b) for a, b in perturbation.lost_contacts}
    for fs in model._contacts:
        if set(fs) & set(perturbation.unavailable_components):
            unavailable_ifaces.add(fs)
    unavailable_ifaces_t = tuple(sorted(tuple(sorted(e)) for e in unavailable_ifaces if len(e) == 2))

    largest = maximal[0] if maximal else ()
    return VariantImpactReport(
        label=perturbation.label,
        wt_orders=wt_orders,
        perturbed_orders=pert_orders,
        wt_trees=wt_trees,
        perturbed_trees=pert_trees,
        full_target_reachable=full,
        largest_reachable_state=largest,
        trapped_intermediates=tuple(maximal) if not full else (),
        first_blocked_transitions=tuple(sorted(blocked)),
        unavailable_interfaces=unavailable_ifaces_t,
        surviving_productive_route=bool(pert_orders > 0 and full),
        provenance=perturbation.provenance,
    )
