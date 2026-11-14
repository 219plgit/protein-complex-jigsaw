"""GFlowNet vs exact flow on the typed neuronal SNARE merger space (1,242 trees)."""
import json
import math
import sys

import numpy as np

from gfn import SplitTable, Sampler
from models import Model, SNARE_CON, SNARE_PRE, nodes

m = Model(SNARE_CON, SNARE_PRE)
TREES = m.trees()
assert len(TREES) == 1242


def crossing(A, B):
    return sum(bin(m.con[i] & B).count("1") for i in range(m.n) if A >> i & 1)


QUERIES = {"SNAP25:Syntaxin-1A (acceptor)": ["SNAP25", "Syntaxin-1A"],
           "VAMP2:SNAP25:Syntaxin-1A (ternary)": ["VAMP2", "SNAP25", "Syntaxin-1A"],
           "VAMP2:Syntaxin-1A": ["VAMP2", "Syntaxin-1A"]}


def rugged(sigma, salt=12345):
    """Synthetic stress-test reward: an independent Gaussian log-potential for
    every admissible merge (A, B), fixed by a seeded generator. Not biological."""
    cache = {}

    def f(A, B):
        key = (min(A, B), max(A, B))
        if key not in cache:
            rng = np.random.default_rng([salt, key[0], key[1]])
            cache[key] = float(sigma * rng.standard_normal())
        return cache[key]
    return f


def tree_logR(t, log_phi):
    if isinstance(t, int):
        return 0.0
    S, a, b = t
    A = a if isinstance(a, int) else a[0]
    B = b if isinstance(b, int) else b[0]
    return log_phi(A, B) + tree_logR(a, log_phi) + tree_logR(b, log_phi)


def evaluate(sampler, log_phi):
    logpi = sampler.all_logpi().detach()
    logq = np.array([sampler.tree_logq(t, logpi) for t in TREES])
    logr = np.array([tree_logR(t, log_phi) for t in TREES])
    logZ = float(np.logaddexp.reduce(logr))
    logp = logr - logZ
    p, q = np.exp(logp), np.exp(logq)
    res = dict(n_trees=len(TREES), q_mass=float(q.sum()),
               logZ_exact=logZ, logZ_learned=float(sampler.policy.logZ),
               TV=float(0.5 * np.abs(p - q).sum()),
               KL_p_q=float((p * (logp - logq)).sum()),
               Neff_exact=float(np.exp(-(p * logp).sum())),
               Neff_learned=float(np.exp(-(q * logq).sum())),
               exact_split_check=float(sampler.t.logF[m.full]))
    for name, q_ in QUERIES.items():
        mask = m.mask(q_)
        ind = np.array([mask in nodes(t) for t in TREES])
        res[f"support_exact[{name}]"] = float(p[ind].sum())
        res[f"support_learned[{name}]"] = float(q[ind].sum())
    return res


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    out = {}
    for label, kind in [("uniform", "u"), ("rugged merge energies, sigma=1.5", "r")]:
        log_phi = {"u": lambda A, B: 0.0, "r": rugged(1.5)}[kind]
        tab = SplitTable(m, log_phi)
        res_seeds = []
        for seed in range(3):
            s = Sampler(tab, seed=seed)
            hist = s.train(steps=steps, batch=64)
            r = evaluate(s, log_phi)
            r["train_seconds"] = hist[-1][3]
            r["seed"] = seed
            res_seeds.append(r)
            print(label, seed, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
        out[label] = res_seeds
    json.dump(out, open("snare_results.json", "w"), indent=1)
