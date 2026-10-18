"""Evidence-gated failure/misassembly assessment.

PCJ never converts failure of productive assembly into an aggregation claim.
Off-pathway states are considered only when the user supplies an explicit rule
with provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

from .variant import VariantImpactReport


@dataclass(frozen=True)
class OffPathwayRule:
    label: str
    requires_exposed: tuple[str, ...] = ()
    requires_trapped_components: tuple[str, ...] = ()
    outcome: str = "off-pathway state"
    provenance: str | None = None

    @property
    def evidence_gated(self) -> bool:
        return bool(self.provenance)


@dataclass
class MisassemblyAssessment:
    productive_failure: bool
    aggregation_inferred: bool
    supported_off_pathway_outcomes: tuple[dict, ...]
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


def assess_misassembly(report: VariantImpactReport,
                       rules: Iterable[OffPathwayRule] = ()) -> MisassemblyAssessment:
    """Evaluate explicit off-pathway rules against a perturbation report.

    The function is deliberately conservative: no rule, or a rule without
    provenance, can produce a supported misassembly outcome.
    """
    productive_failure = not report.full_target_reachable
    trapped = set(report.largest_reachable_state)
    exposed = {x for edge in report.unavailable_interfaces for x in edge}
    supported = []

    for rule in rules:
        if not rule.evidence_gated:
            continue
        if not set(rule.requires_exposed).issubset(exposed):
            continue
        if not set(rule.requires_trapped_components).issubset(trapped):
            continue
        supported.append({
            "label": rule.label,
            "outcome": rule.outcome,
            "provenance": rule.provenance,
        })

    return MisassemblyAssessment(
        productive_failure=productive_failure,
        aggregation_inferred=False,
        supported_off_pathway_outcomes=tuple(supported),
        note=("Assembly failure alone is not evidence of aggregation. "
              "Off-pathway outcomes are returned only from explicit, "
              "provenance-bearing rules."),
    )
