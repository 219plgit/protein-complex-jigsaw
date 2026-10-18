"""PCJ command-line additions."""
from __future__ import annotations

import argparse
import json

from scijigsaw.cases import ALL_CASES
from .variant import Perturbation, analyse_perturbation


def variant_main(argv=None):
    ap = argparse.ArgumentParser(
        description="Protein Complex Jigsaw variant-to-mechanism analysis")
    ap.add_argument("case", choices=sorted(ALL_CASES))
    ap.add_argument("--label", required=True)
    ap.add_argument("--unavailable", action="append", default=[],
                    help="component rendered unavailable; repeat as needed")
    ap.add_argument("--lost-contact", nargs=2, action="append", default=[],
                    metavar=("A", "B"), help="declared lost interface")
    ap.add_argument("--exclude", nargs=2, action="append", default=[],
                    metavar=("A", "B"), help="declared new exclusion")
    ap.add_argument("--provenance", default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    p = Perturbation(
        label=a.label,
        unavailable_components=tuple(a.unavailable),
        lost_contacts=tuple(tuple(x) for x in a.lost_contact),
        added_exclusions=tuple(tuple(x) for x in a.exclude),
        provenance=a.provenance,
    )
    r = analyse_perturbation(ALL_CASES[a.case], p)
    if a.json:
        print(json.dumps(r.to_dict(), indent=2))
    else:
        print(f"Protein Complex Jigsaw — {r.label}")
        print(f"WT permitted orders: {r.wt_orders}")
        print(f"Perturbed permitted orders: {r.perturbed_orders}")
        print(f"Full target reachable: {r.full_target_reachable}")
        print("Largest reachable state:", ", ".join(r.largest_reachable_state) or "—")
        if r.first_blocked_transitions:
            print("First blocked transitions:")
            for state, unit, reason in r.first_blocked_transitions:
                print(f"  {state} -> {unit}: {reason}")
