"""Amortised GFlowNet (trajectory balance) on PCJ merger-tree spaces, compared
with the exact flow.

Generative process (top-down, unique trajectory per tree): start from the full
complex; repeatedly take the most recently created unresolved block (LIFO) and
choose one admissible canonical split {A, B}. Each complete merger tree has
exactly one trajectory, so the backward policy is 1 and trajectory balance is
    (log Z + sum_t log pi(a_t | s_t) - log R(x))^2.
The policy scores each admissible split of the block being split with an MLP on
the one-hot block / part masks; it does not see the exact flow.
"""
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)


def popcount(x):
    return bin(x).count("1")


class SplitTable:
    """All blocks reachable from the full complex and their admissible splits,
    with exact flow under a per-merge potential log phi(A, B)."""

    def __init__(self, model, log_phi):
        self.m = model
        self.log_phi = log_phi
        self.blocks = {}          # block mask -> list of (A, B)
        self._collect(model.full)
        self.logF = {}
        self._exact(model.full)
        # mask dead-end actions: keep only splits whose parts can be completed
        ok = lambda X: X & (X - 1) == 0 or self.logF.get(X, -math.inf) > -math.inf
        pruned, todo = {}, [model.full]
        while todo:
            S = todo.pop()
            if S in pruned or S & (S - 1) == 0:
                continue
            pruned[S] = [(A, B) for A, B in self.blocks[S] if ok(A) and ok(B)]
            for A, B in pruned[S]:
                todo += [A, B]
        self.blocks = pruned

    def _collect(self, S):
        if S in self.blocks or S & (S - 1) == 0:
            return
        sp = list(self.m.splits(S))
        self.blocks[S] = sp
        for A, B in sp:
            self._collect(A)
            self._collect(B)

    def _exact(self, S):
        if S & (S - 1) == 0:
            return 0.0
        if S in self.logF:
            return self.logF[S]
        terms = [self.log_phi(A, B) + self._exact(A) + self._exact(B)
                 for A, B in self.blocks[S]]
        v = float(np.logaddexp.reduce(terms)) if terms else -math.inf
        self.logF[S] = v
        return v

    def exact_logp_split(self, S, A, B):
        return self.log_phi(A, B) + self.logF.get(A, 0.0) + self.logF.get(B, 0.0) - self.logF[S]


class Policy(nn.Module):
    def __init__(self, n, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * n, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        self.logZ = nn.Parameter(torch.zeros(()))

    def forward(self, feats):
        return self.net(feats).squeeze(-1)


class Sampler:
    def __init__(self, table, hidden=128, seed=0):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.t = table
        n = table.m.n
        self.n = n
        self.block_list = list(table.blocks)
        feats, owner, parts = [], [], []
        for bi, S in enumerate(self.block_list):
            for A, B in table.blocks[S]:
                feats.append([(S >> i) & 1 for i in range(n)] +
                             [(A >> i) & 1 for i in range(n)] +
                             [(B >> i) & 1 for i in range(n)])
                owner.append(bi)
                parts.append((A, B))
        self.feats = torch.tensor(feats, dtype=torch.float32)
        self.owner = torch.tensor(owner)
        self.parts = parts
        self.block_index = {S: i for i, S in enumerate(self.block_list)}
        # candidate index ranges per block
        self.ranges = []
        start = 0
        for S in self.block_list:
            k = len(table.blocks[S])
            self.ranges.append((start, start + k))
            start += k
        self.policy = Policy(n, hidden)

    def all_logpi(self):
        """log pi(split | block) for every candidate, via a segmented softmax."""
        logits = self.policy(self.feats)
        nb = len(self.block_list)
        mx = torch.full((nb,), -1e30).scatter_reduce(0, self.owner, logits.detach(), reduce="amax")
        z = torch.zeros(nb).scatter_add(0, self.owner, torch.exp(logits - mx[self.owner]))
        return logits - mx[self.owner] - torch.log(z)[self.owner]

    def sample(self, logpi, eps=0.0):
        """One trajectory: returns list of candidate indices and the tree's log R."""
        stack = [self.t.m.full]
        chosen, logR = [], 0.0
        lp = logpi.detach().numpy()
        while stack:
            S = stack.pop()
            if S & (S - 1) == 0:
                continue
            a, b = self.ranges[self.block_index[S]]
            if eps > 0 and random.random() < eps:
                c = random.randrange(a, b)
            else:
                p = np.exp(lp[a:b] - lp[a:b].max())
                c = a + int(np.random.choice(b - a, p=p / p.sum()))
            chosen.append(c)
            A, B = self.parts[c]
            logR += self.t.log_phi(A, B)
            stack.append(A)
            stack.append(B)
        return chosen, logR

    def _logpi_subset(self, block_ids):
        """Differentiable log pi for all candidates of the given blocks only."""
        idx = torch.cat([torch.arange(*self.ranges[b]) for b in block_ids])
        local_owner = torch.cat([torch.full((self.ranges[b][1] - self.ranges[b][0],), k)
                                 for k, b in enumerate(block_ids)])
        logits = self.policy(self.feats[idx])
        nb = len(block_ids)
        mx = torch.full((nb,), -1e30).scatter_reduce(0, local_owner, logits.detach(), reduce="amax")
        z = torch.zeros(nb).scatter_add(0, local_owner, torch.exp(logits - mx[local_owner]))
        lp = logits - mx[local_owner] - torch.log(z)[local_owner]
        pos = {int(g): i for i, g in enumerate(idx)}
        return lp, pos

    def train(self, steps=2000, batch=64, lr=1e-3, lr_z=0.1, eps0=0.1, log_every=500):
        opt = torch.optim.Adam([{"params": self.policy.net.parameters(), "lr": lr},
                                {"params": [self.policy.logZ], "lr": lr_z}])
        hist = []
        t0 = time.time()
        for step in range(steps):
            eps = eps0 * max(0.0, 1 - step / (0.7 * steps))
            with torch.no_grad():
                logpi_all = self.all_logpi()
            trajs = [self.sample(logpi_all, eps) for _ in range(batch)]
            blocks = sorted({int(self.owner[c]) for ch, _ in trajs for c in ch})
            lp, pos = self._logpi_subset(blocks)
            losses = []
            for ch, logR in trajs:
                sel = torch.tensor([pos[c] for c in ch])
                losses.append((self.policy.logZ + lp[sel].sum() - logR) ** 2)
            loss = torch.stack(losses).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (step + 1) % log_every == 0 or step == 0 or step == steps - 1:
                hist.append((step + 1, float(loss.detach()), float(self.policy.logZ.detach()),
                             time.time() - t0))
        return hist

    # --------------------------------------------------- exact evaluation
    def tree_logq(self, tree, logpi):
        """log probability of an explicit tree under the learned policy."""
        if isinstance(tree, int):
            return 0.0
        S, ta, tb = tree
        A = ta if isinstance(ta, int) else ta[0]
        B = tb if isinstance(tb, int) else tb[0]
        if not (A & (S & -S)):
            A, B, ta, tb = B, A, tb, ta
        a, b = self.ranges[self.block_index[S]]
        c = next(i for i in range(a, b) if self.parts[i] == (A, B))
        return float(logpi[c]) + self.tree_logq(ta, logpi) + self.tree_logq(tb, logpi)
