"""Figure: amortised GFlowNet against the exact flow.
A: SNARE merger space, synthetic rugged reward - exact vs learned probability of
   every one of the 1,242 trees (seed 0, retrained here with the same settings).
B: n=14 synthetic space (3.6e8 trees) - exact vs estimated support of every
   contact-pair subcomplex, learned sampler (3 seeds) and exact sampler.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gfn import SplitTable, Sampler
from run_snare import m, TREES, rugged, tree_logR

f = rugged(1.5)
tab = SplitTable(m, f)
s = Sampler(tab, seed=0)
s.train(steps=3000, batch=64)
logpi = s.all_logpi().detach()
logq = np.array([s.tree_logq(t, logpi) for t in TREES])
logr = np.array([tree_logR(t, f) for t in TREES])
logp = logr - np.logaddexp.reduce(logr)
np.savez("fig_snare_arrays.npz", logp=logp, logq=logq)

L = json.load(open("large_results.json"))
key = "rugged merge energies, sigma=1.5"

INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
C1, C2 = "#2563eb", "#d97706"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))

a = ax[0]
p, q = np.exp(logp), np.exp(logq)
lo = min(p.min(), q.min()) * 0.5
a.plot([lo, 1], [lo, 1], color=GRID, lw=1, zorder=0)
a.scatter(p, q, s=7, color=C1, alpha=0.6, linewidths=0)
a.set_xscale("log"); a.set_yscale("log")
a.set_xlim(lo, 1); a.set_ylim(lo, 1)
a.set_xlabel("exact probability of tree")
a.set_ylabel("learned probability of tree")
tv = 0.5 * np.abs(p - q).sum()
a.set_title(f"A  SNARE, 1,242 trees (TV = {tv:.3f})", loc="left", fontsize=9, color=INK)
for sp in ("top", "right"):
    a.spines[sp].set_visible(False)

b = ax[1]
ex = np.array(L[key]["support_exact"])
b.plot([0, 1], [0, 1], color=GRID, lw=1, zorder=0)
for r in L[key]["runs"]:
    b.scatter(ex, r["support_learned"], s=12, color=C1, alpha=0.55, linewidths=0,
              label="amortised GFlowNet" if r["seed"] == 0 else None)
b.scatter(ex, L[key]["support_exact_sampler"], s=14, facecolors="none", edgecolors=C2,
          linewidths=0.9, label="exact sampler")
b.set_xlabel("exact subcomplex support")
b.set_ylabel("estimated support (10,000 samples)")
b.set_title("B  synthetic, n = 14, 3.6×10$^8$ trees", loc="left", fontsize=9, color=INK)
b.legend(frameon=False, fontsize=8, loc="upper left")
for sp in ("top", "right"):
    b.spines[sp].set_visible(False)

fig.tight_layout()
fig.savefig("fig_gflownet.pdf")
fig.savefig("fig_gflownet.png", dpi=200)
print("TV", tv)
