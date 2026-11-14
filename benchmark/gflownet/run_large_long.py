"""Rugged synthetic space: longer training with more exploration."""
import json, math, sys
import numpy as np
from gfn import SplitTable, Sampler
from run_large import random_model, exact_support_log, sample_learned
from run_snare import rugged

steps, eps0, nsamp = int(sys.argv[1]), float(sys.argv[2]), 10000
m, edges, pre = random_model()
pairs = [m.mask([a, b]) for a, b in edges]
tab = SplitTable(m, rugged(1.5))
exact_sup = np.array([exact_support_log(tab, E) for E in pairs])
logZ = tab.logF[m.full]
runs = []
for seed in range(3):
    s = Sampler(tab, seed=seed)
    hist = s.train(steps=steps, batch=64, eps0=eps0)
    logpi = s.all_logpi().detach()
    smp = [sample_learned(s, logpi) for _ in range(nsamp)]
    ln = np.array([np.mean([E in set(nd) for nd, _, _ in smp]) for E in pairs])
    r = dict(seed=seed, steps=steps, eps0=eps0, logZ_exact=logZ, logZ_learned=float(s.policy.logZ.detach()),
             KL_q_p=float(np.mean([lq - (lr - logZ) for _, lq, lr in smp])),
             support_MAE=float(np.abs(ln - exact_sup).mean()), support_maxAE=float(np.abs(ln - exact_sup).max()),
             train_seconds=hist[-1][3])
    print(r, flush=True); runs.append(r)
json.dump(runs, open(f"large_long_{steps}_{eps0}.json", "w"), indent=1)
