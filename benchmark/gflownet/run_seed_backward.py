"""GFlowNet with a non-trivial backward policy on the seed-anchored state graph.

The object is the assembled complex (single terminal); trajectories are the
seed-anchored histories; states are SETS of placed components, so states have
several parents and the backward policy P_B is not 1.

Two regimes, both compared with exact flows from pcjigsaw.backward:

  fixed P_B   P_B encodes disassembly (uniform, or a rugged synthetic stress test).
              Trajectory balance with log P_B terms; the learned forward policy
              should reproduce the exact pathway distribution prod P_B.
  learned P_B P_B is a second network trained jointly, as in standard GFlowNets.
              The terminal flow is still identified (log Z -> 0), but the pathway
              distribution is not: it depends on the run. This is why P_B must be
              fixed by evidence if it is to carry biological meaning.

    python run_seed_backward.py [steps]
"""
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from pcjigsaw.prob import TypedModel                                     # noqa: E402
from pcjigsaw import backward as B                                       # noqa: E402

torch.set_num_threads(2)

# the paper's 252-history VAMP2 seating model (typed SNARE encoding + SNAP25 < syntaxin-1A)
CON = [("VAMP2", "SNAP25"), ("VAMP2", "Syntaxin-1A"), ("VAMP2", "Complexin"),
       ("SNAP25", "Syntaxin-1A"), ("SNAP25", "Syt-1"), ("Syntaxin-1A", "Complexin"),
       ("Syntaxin-1A", "Syt-1"), ("Syntaxin-1A", "Munc18-1"),
       ("VAMP2", "Synaptophysin"), ("VAMP2", "SNCA")]
PRE = [("VAMP2", "Complexin"), ("Syntaxin-1A", "Complexin"), ("SNAP25", "Syt-1"),
       ("Syntaxin-1A", "Syt-1"), ("Syntaxin-1A", "Munc18-1"), ("SNAP25", "Syntaxin-1A")]
SEED = "VAMP2"
MODEL = TypedModel(V=tuple(sorted({u for e in CON for u in e})),
                   C=frozenset(frozenset(e) for e in CON), P=tuple(PRE))
UNITS = [v for v in MODEL.V if v != SEED]
IDX = {u: i for i, u in enumerate(UNITS)}
STATES = B.reachable_states(MODEL, SEED)
FULL = max(STATES, key=len)


def rugged_backward(sigma=1.5, salt=7):
    """Synthetic stress test: an independent Gaussian log-propensity for every
    admissible dissociation (S, p). Not a biological model."""
    def pb(S):
        ps = B.parents(MODEL, SEED, S, STATES)
        w = {}
        for p in ps:
            key = sum(1 << IDX[u] for u in S)          # deterministic across processes
            rng = np.random.default_rng([salt, key, IDX[p]])
            w[p] = math.exp(sigma * rng.standard_normal())
        z = sum(w.values())
        return {p: v / z for p, v in w.items()}
    return pb


def enc(S):
    x = torch.zeros(len(UNITS))
    for u in S:
        x[IDX[u]] = 1.0
    return x


def legal_add(S):
    return [p for p in UNITS if p not in S and (S | {p}) in STATE_SET
            and p in B.parents(MODEL, SEED, S | {p}, STATES)]


STATE_SET = set(STATES)
LEGAL = {S: legal_add(S) for S in STATES}
PARENTS = {S: B.parents(MODEL, SEED, S, STATES) for S in STATES}


class Net(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(len(UNITS), h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                               nn.Linear(h, len(UNITS)))

    def logp(self, S, allowed):
        z = self.f(enc(S))
        mask = torch.full((len(UNITS),), -1e9)
        for p in allowed:
            mask[IDX[p]] = 0.0
        return torch.log_softmax(z + mask, dim=0)


def train(pb_fixed, steps, seed, learn_pb=False, batch=32, eps=0.05, lr=3e-3):
    torch.manual_seed(seed); random.seed(seed)
    fwd = Net(); bwd = Net() if learn_pb else None
    logZ = nn.Parameter(torch.zeros(()))
    params = list(fwd.parameters()) + [logZ] + (list(bwd.parameters()) if learn_pb else [])
    opt = torch.optim.Adam(params, lr=lr)
    t0 = time.time()
    for it in range(steps):
        loss = 0.0
        for _ in range(batch):
            S = frozenset(); lpf = 0.0; lpb = 0.0
            while S != FULL:
                acts = LEGAL[S]
                lp = fwd.logp(S, acts)
                if random.random() < eps:
                    p = random.choice(acts)
                else:
                    p = UNITS[torch.multinomial(lp.exp(), 1).item()]
                lpf = lpf + lp[IDX[p]]
                T = S | {p}
                if learn_pb:
                    lpb = lpb + bwd.logp(T, PARENTS[T])[IDX[p]]
                else:
                    lpb = lpb + math.log(pb_fixed(T)[p])
                S = T
            loss = loss + (logZ + lpf - lpb) ** 2          # R(terminal) = 1
        opt.zero_grad(); (loss / batch).backward(); opt.step()
    return fwd, bwd, logZ.item(), time.time() - t0


def learned_distribution(fwd):
    H = B.histories(MODEL, SEED)
    out = {}
    with torch.no_grad():
        for h in H:
            S = frozenset(); lp = 0.0
            for p in h:
                lp += float(fwd.logp(S, LEGAL[S])[IDX[p]]); S = S | {p}
            out[h] = math.exp(lp)
    z = sum(out.values())
    return {h: v / z for h, v in out.items()}, z


def tv(p, q):
    return 0.5 * sum(abs(p[h] - q.get(h, 0.0)) for h in p)


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    only = sys.argv[2] if len(sys.argv) > 2 else None     # e.g. "rugged": rerun one regime, keep the rest
    out = os.path.join(HERE, "results", "seed_backward_results.json")
    if only:
        res = json.load(open(out))
        name, pb = "rugged disassembly, sigma=1.5", rugged_backward()
        target = B.pathway_distribution(MODEL, SEED, pb)
        rows = []
        for seed in range(3):
            fwd, _, lz, sec = train(pb, steps, seed)
            q, mass = learned_distribution(fwd)
            rows.append({"seed": seed, "TV": tv(target, q), "logZ": lz, "mass_on_admissible": mass,
                         "Neff_exact": B.effective_number(target), "Neff_learned": B.effective_number(q),
                         "seconds": sec})
            print(name, rows[-1], flush=True)
        res["runs"][name] = rows
        json.dump(res, open(out, "w"), indent=1)
        sys.exit(0)
    multi = sum(len(PARENTS[S]) > 1 for S in STATES)
    res = {"model": "VAMP2 seating (252 histories)", "states": len(STATES),
           "states_with_several_parents": multi, "steps": steps, "runs": {}}
    for name, pb in [("uniform disassembly", B.uniform_backward(MODEL, SEED)),
                     ("rugged disassembly, sigma=1.5", rugged_backward())]:
        target = B.pathway_distribution(MODEL, SEED, pb)
        rows = []
        for seed in range(3):
            fwd, _, lz, sec = train(pb, steps, seed)
            q, mass = learned_distribution(fwd)
            rows.append({"seed": seed, "TV": tv(target, q), "logZ": lz, "mass_on_admissible": mass,
                         "Neff_exact": B.effective_number(target), "Neff_learned": B.effective_number(q),
                         "seconds": sec})
            print(name, rows[-1], flush=True)
        res["runs"][name] = rows
    # learned P_B: terminal flow identified, pathway distribution not
    dists, rows = [], []
    for seed in range(3):
        fwd, bwd, lz, sec = train(None, steps, seed, learn_pb=True)
        q, mass = learned_distribution(fwd)
        dists.append(q)
        rows.append({"seed": seed, "logZ": lz, "Neff_learned": B.effective_number(q), "seconds": sec})
        print("learned P_B", rows[-1], flush=True)
    pair = [tv(dists[i], dists[j]) for i in range(3) for j in range(i + 1, 3)]
    res["runs"]["learned P_B"] = {"runs": rows, "pairwise_TV_between_runs": pair}
    print("learned P_B pairwise TV between runs:", pair)
    with open(os.path.join(HERE, "results", "seed_backward_results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
