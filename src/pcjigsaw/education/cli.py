"""pcjigsaw-teach: generate teaching material from the research model."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="pcjigsaw-teach",
        description="Generate teaching material from the same executable model used for research. "
                    "Every answer is computed, not typed.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("vamp2", help="the four-step VAMP2 activity, tile sets and advanced PCJ Flow tasks")
    v.add_argument("out_dir")
    v.add_argument("--no-kits", action="store_true", help="skip the printable tile sets")

    p = sub.add_parser("pack", help="outreach and undergraduate pack for a small contact model")
    p.add_argument("model", help="built-in model (eif3, chain4, cycle4) or a directory with "
                                 "proteins.csv and interactions.csv")
    p.add_argument("out_dir")
    p.add_argument("--seed")
    p.add_argument("--query", help="comma-separated pair, e.g. Tif34,Tif35")
    p.add_argument("--no-kits", action="store_true")

    c = sub.add_parser("check", help="print the VAMP2 and eIF3 answer-key numbers and verify them against the paper")

    a = ap.parse_args(argv)
    if a.cmd == "vamp2":
        from .vamp2 import build
        R = build(a.out_dir, kits=not a.no_kits)
        print(f"VAMP2: {R['orders_admissible']} of {R['orders_total']} orders; 536-tree merger space: "
              f"{R['merger_trees_typed']}; 1KIL revision {R['revision_1KIL']['before']} -> "
              f"{R['revision_1KIL']['after']}  -> {a.out_dir}")
    elif a.cmd == "pack":
        from .teach import build_pack
        R = build_pack(a.model, a.out_dir, a.seed, a.query.split(",") if a.query else None, not a.no_kits)
        print(f"{R['model']}: {R['seeded']['count']} seed-anchored histories, {R['merger']['count']} merger trees, "
              f"query {'-'.join(R['query'])} {R['merger']['query_support'][0]}/{R['merger']['count']}  -> {a.out_dir}")
    else:
        from .vamp2 import compute_vamp2
        from .teach import TeachingModel, compute, model_path
        V = compute_vamp2()
        E = compute(TeachingModel.from_dir(model_path("eif3")))
        got = {"VAMP2 seed-anchored": V["orders_admissible"], "VAMP2 unconstrained": V["orders_total"],
               "VAMP2 merger trees": V["merger_trees_typed"], "VAMP2 before 1KIL": V["revision_1KIL"]["before"],
               "VAMP2 core histories": next(q["histories_through_state"] for q in V["queries"]
                                            if q["state"] == ["VAMP2", "SNAP25", "Syntaxin-1A"]),
               "eIF3 seed-anchored": E["seeded"]["count"], "eIF3 merger trees": E["merger"]["count"],
               "eIF3 Tif34-Tif35 in merger trees": E["merger"]["query_support"][0],
               "reward keeps all histories": V["reward"]["admissible_under_all_checked_priors"],
               "reward rescaling invariant": V["reward"]["rescaling_invariant"]}
        want = {"VAMP2 seed-anchored": 252, "VAMP2 unconstrained": 5040, "VAMP2 merger trees": 536,
                "VAMP2 before 1KIL": 336, "VAMP2 core histories": 120, "eIF3 seed-anchored": 6,
                "eIF3 merger trees": 14, "eIF3 Tif34-Tif35 in merger trees": 5,
                "reward keeps all histories": True, "reward rescaling invariant": True}
        bad = 0
        for k, w in want.items():
            ok = got[k] == w
            bad += not ok
            print(f"[{'ok' if ok else 'FAIL'}] {k}: {got[k]} (paper: {w})")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
