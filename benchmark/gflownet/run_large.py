"""GFlowNet vs exact flow on a larger synthetic merger space (too large to
enumerate tree by tree, but exact by dynamic programming)."""
import json
import math
import random
import sys
import time

import numpy as np
import torch

from gfn import SplitTable, Sampler
from models import Model
from run_snare import rugged


def random_model(n=14, n_edges=22, n_pre=4, seed=7):
    rng = random.Random(seed)
    names = [f"P{i:02d}" for i in range(n)]
    order = names[:]
    rng.shuffle(order)
    edges = set()
    for i in range(1, n):                      # random spanning tree
        j = rng.randrange(i)
        edges.add(tuple(sorted((order[i], order[j]))))
    while len(edges) < n_edges:
        a, b = rng.sample(names, 2)
        edges.add(tuple(sorted((a, b))))
    edges = sorted(edges)
    rank = {u: k for k, u in enumerate(order)}
    pre = []
    for a, b in rng.sample(edges, n_pre):      # orient along a fixed order: acyclic
        pre.append((a, b) if rank[a] < rank[b] else (b, a))
    return Model(edges, pre), edges, pre


def exact_support_log(tab, E):
    m, memo = tab.m, {}

    def G(S):
        if S == E:
            return tab.logF.get(S, 0.0)
        if E & ~S or S & (S - 1) == 0:
            return -math.inf
        if S in memo:
            return memo[S]
        terms = []
        for A, B in tab.blocks.get(S, []):
            if not (E & ~A):
                terms.append(tab.log_phi(A, B) + G(A) + tab.logF.get(B, 0.0))
            elif not (E & ~B):
                terms.append(tab.log_phi(A, B) + tab.logF.get(A, 0.0) + G(B))
        v = float(np.logaddexp.reduce(terms)) if terms else -math.inf
        memo[S] = v
        return v
    return math.exp(G(m.full) - tab.logF[m.full])


def sample_exact(tab, rng):
    stack, nodes_, logR = [tab.m.full], [], 0.0
    while stack:
        S = stack.pop()
        nodes_.append(S)
        if S & (S - 1) == 0:
            continue
        sp = tab.blocks[S]
        lp = np.array([tab.exact_logp_split(S, A, B) for A, B in sp])
        p = np.exp(lp - lp.max()); p /= p.sum()
        A, B = sp[rng.choice(len(sp), p=p)]
        logR += tab.log_phi(A, B)
        stack += [A, B]
    return nodes_, logR


def sample_learned(s, logpi):
    stack, nodes_, logq, logR = [s.t.m.full], [], 0.0, 0.0
    lp = logpi.numpy()
    while stack:
        S = stack.pop()
        nodes_.append(S)
        if S & (S - 1) == 0:
            continue
        a, b = s.ranges[s.block_index[S]]
        p = np.exp(lp[a:b] - lp[a:b].max()); p /= p.sum()
        c = a + int(np.random.choice(b - a, p=p))
        logq += lp[c]
        A, B = s.parts[c]
        logR += s.t.log_phi(A, B)
        stack += [A, B]
    return nodes_, logq, logR


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    nsamp = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    m, edges, pre = random_model()
    pairs = [m.mask([a, b]) for a, b in edges]
    out = dict(n=m.n, edges=edges, prerequisites=pre)
    for label, log_phi in [("uniform", lambda A, B: 0.0),
                           ("rugged merge energies, sigma=1.5", rugged(1.5))]:
        t0 = time.time()
        tab = SplitTable(m, log_phi)
        n_cand = sum(len(v) for v in tab.blocks.values())
        exact_sup = np.array([exact_support_log(tab, E) for E in pairs])
        logZ = tab.logF[m.full]
        t_exact = time.time() - t0
        rng = np.random.default_rng(0)
        ex_nodes = [set(sample_exact(tab, rng)[0]) for _ in range(nsamp)]
        ex_sup = np.array([np.mean([E in nd for nd in ex_nodes]) for E in pairs])
        runs = []
        for seed in range(3):
            s = Sampler(tab, seed=seed)
            hist = s.train(steps=steps, batch=64)
            logpi = s.all_logpi().detach()
            samples = [sample_learned(s, logpi) for _ in range(nsamp)]
            kl_qp = float(np.mean([lq - (lr - logZ) for _, lq, lr in samples]))
            ln_sup = np.array([np.mean([E in set(nd) for nd, _, _ in samples]) for E in pairs])
            r = dict(seed=seed, logZ_learned=float(s.policy.logZ.detach()),
                     KL_q_p=kl_qp, support_MAE=float(np.abs(ln_sup - exact_sup).mean()),
                     support_maxAE=float(np.abs(ln_sup - exact_sup).max()),
                     train_seconds=hist[-1][3], support_learned=ln_sup.tolist())
            runs.append(r)
            print(label, {k: v for k, v in r.items() if k != 'support_learned'}, flush=True)
        res = dict(log_trees_or_Z=logZ, n_blocks=len(tab.blocks), n_candidate_splits=n_cand,
                   exact_seconds=t_exact,
                   exact_sampler_support_MAE=float(np.abs(ex_sup - exact_sup).mean()),
                   exact_sampler_support_maxAE=float(np.abs(ex_sup - exact_sup).max()),
                   support_exact=exact_sup.tolist(), support_exact_sampler=ex_sup.tolist(),
                   runs=runs)
        if label == "uniform":
            res["n_trees"] = int(round(math.exp(logZ)))
        out[label] = res
        print(label, {k: v for k, v in res.items() if k not in ('runs', 'support_exact', 'support_exact_sampler')}, flush=True)
    json.dump(out, open("large_results.json", "w"), indent=1)
