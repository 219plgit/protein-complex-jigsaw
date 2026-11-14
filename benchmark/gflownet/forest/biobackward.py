"""Minimal multi-parent GFlowNet for PCJ merger trees.

Bottom-up construction keeps an unordered forest of partial merger trees. Any
compatible pair of trees can be merged, so independent merges can be interleaved
in different orders and a state can have multiple parents.

The backward policy is deliberately minimal.  For each reversible root merge in
the current forest, define a biological relation count

    b = (# admitted contacts crossing the merge)
        + (# admitted prerequisite arcs crossing the merge).

The reverse logit is -beta*b, so beta=0 is the uniform backward policy and
beta>0 preferentially preserves merges supported by more admitted relations.
This is reverse credit assignment, not a claim of kinetics or evidence truth.

Two properties limit what this construction can say about biology (see
Supplementary Section S11.5 and tests/test_forest_backward.py):
  * the terminal objects are merger trees, and for every beta the backward
    probabilities of each tree's trajectories sum to 1, so at the optimum every
    tree has probability R(x)/Z whatever beta is: beta changes only the timing of
    independent merges, never the support of a subcomplex;
  * beta is therefore not identified by trajectory balance. It is fixed and
    swept by default; learn_backward=True is kept only for experiments.
Contributed for the v12 revision; the exact engine in exact.py is cross-checked
against pcjigsaw.prob.merger_F by the tests.
"""
from __future__ import annotations
import math, random, time
from functools import lru_cache
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)


def mask_of(t):
    return t if isinstance(t, int) else t[0]


def canon_tree(A, B, ta, tb):
    # same canonical orientation as Model.splits: left contains lowest bit
    S = A | B
    low = S & -S
    if not (A & low):
        A, B, ta, tb = B, A, tb, ta
    return (S, ta, tb)


def tree_key(t):
    if isinstance(t, int):
        return (0, t)
    return (1, t[0], tree_key(t[1]), tree_key(t[2]))


def canon_forest(fs):
    return tuple(sorted(fs, key=tree_key))


class MergePolicy(nn.Module):
    def __init__(self, n, hidden=128, learn_backward=False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3*n, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        self.logZ = nn.Parameter(torch.zeros(()))
        # softplus(0) = 0.693: modest relation-aware initial backward policy.
        self.raw_beta = nn.Parameter(torch.tensor(0.0), requires_grad=learn_backward)

    @property
    def beta(self):
        return F.softplus(self.raw_beta)

    def forward(self, feats):
        return self.net(feats).squeeze(-1)


class ForestSampler:
    def __init__(self, model, log_phi, hidden=128, seed=0, learn_backward=False,
                 fixed_beta=1.0):
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        self.m=model; self.log_phi=log_phi; self.n=model.n
        self.policy=MergePolicy(model.n, hidden, learn_backward=learn_backward and fixed_beta is None)
        self.fixed_beta=fixed_beta
        self._complete_cache={}

    def initial(self):
        return canon_forest(tuple(1 << i for i in range(self.n)))

    def _masks(self, forest):
        return tuple(sorted(mask_of(t) for t in forest))

    @lru_cache(None)
    def _can_complete_masks(self, masks):
        masks=tuple(masks)
        if len(masks)==1:
            return masks[0]==self.m.full
        L=list(masks)
        for i in range(len(L)):
            for j in range(i+1,len(L)):
                A,B=L[i],L[j]
                if self.m.merge_ok(A,B):
                    new=tuple(sorted([L[k] for k in range(len(L)) if k not in (i,j)] + [A|B]))
                    if self._can_complete_masks(new): return True
        return False

    def candidates(self, forest):
        out=[]
        L=list(forest)
        for i in range(len(L)):
            for j in range(i+1,len(L)):
                ta,tb=L[i],L[j]; A,B=mask_of(ta),mask_of(tb)
                if not self.m.merge_ok(A,B):
                    continue
                newtree=canon_tree(A,B,ta,tb)
                nf=canon_forest(tuple(L[k] for k in range(len(L)) if k not in (i,j))+(newtree,))
                if self._can_complete_masks(self._masks(nf)):
                    out.append((i,j,A,B,newtree,nf))
        return out

    def feat(self, A,B):
        S=A|B
        return [(S>>i)&1 for i in range(self.n)]+[(A>>i)&1 for i in range(self.n)]+[(B>>i)&1 for i in range(self.n)]

    def forward_logprobs(self, forest):
        cand=self.candidates(forest)
        if not cand: return cand, torch.empty(0)
        feats=torch.tensor([self.feat(c[2],c[3]) for c in cand],dtype=torch.float32)
        return cand, torch.log_softmax(self.policy(feats),dim=0)

    def contact_crossing(self,A,B):
        return sum((self.m.con[i] & B).bit_count() for i in range(self.n) if (A>>i)&1)

    def prereq_crossing(self,A,B):
        # directed admitted prerequisite arcs whose endpoints lie on opposite sides
        c=0
        for v in range(self.n):
            pv=self.m.pre[v]
            if (B>>v)&1: c += (pv & A).bit_count()
            if (A>>v)&1: c += (pv & B).bit_count()
        return c

    def relation_count(self,A,B):
        return self.contact_crossing(A,B)+self.prereq_crossing(A,B)

    def backward_candidates(self, forest):
        """Parents obtained by retracting one root merge of a current forest tree."""
        out=[]
        L=list(forest)
        for i,t in enumerate(L):
            if isinstance(t,int): continue
            S,ta,tb=t; A,B=mask_of(ta),mask_of(tb)
            pf=canon_forest(tuple(L[k] for k in range(len(L)) if k!=i)+(ta,tb))
            out.append((i,A,B,pf,self.relation_count(A,B)))
        return out

    def backward_logprobs(self, forest):
        bc=self.backward_candidates(forest)
        if not bc: return bc, torch.empty(0)
        counts=torch.tensor([x[4] for x in bc],dtype=torch.float32)
        beta=torch.tensor(float(self.fixed_beta)) if self.fixed_beta is not None else self.policy.beta
        logits=-beta*counts
        return bc, torch.log_softmax(logits,dim=0)

    def sample(self, eps=0.0):
        forest=self.initial(); logpf=[]; logpb=[]; logR=0.0
        while len(forest)>1:
            cand,lp=self.forward_logprobs(forest)
            if not cand: raise RuntimeError('dead end')
            if eps>0 and random.random()<eps:
                k=random.randrange(len(cand))
            else:
                p=torch.softmax(lp.detach(),0).cpu().numpy(); k=int(np.random.choice(len(cand),p=p))
            _,_,A,B,newtree,nf=cand[k]
            # backward probability of returning to the current forest from nf
            bc,lb=self.backward_logprobs(nf)
            bk=next(q for q,x in enumerate(bc) if x[3]==forest)
            logpf.append(lp[k]); logpb.append(lb[bk])
            logR += self.log_phi(A,B)
            forest=nf
        return forest[0], logpf, logpb, logR

    def train(self, steps=2000,batch=64,lr=1e-3,lr_z=0.1,lr_beta=2e-3,eps0=0.1,log_every=500):
        groups=[{'params':self.policy.net.parameters(),'lr':lr},{'params':[self.policy.logZ],'lr':lr_z}]
        if self.fixed_beta is None and self.policy.raw_beta.requires_grad:
            groups.append({'params':[self.policy.raw_beta],'lr':lr_beta})
        opt=torch.optim.Adam(groups)
        hist=[]; t0=time.time()
        for step in range(steps):
            eps=eps0*max(0.0,1-step/(0.7*steps))
            losses=[]
            for _ in range(batch):
                _,lpf,lpb,logR=self.sample(eps)
                r=self.policy.logZ+torch.stack(lpf).sum()-logR-torch.stack(lpb).sum()
                losses.append(r*r)
            loss=torch.stack(losses).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step==0 or (step+1)%log_every==0 or step==steps-1:
                beta=float(self.fixed_beta if self.fixed_beta is not None else self.policy.beta.detach())
                hist.append((step+1,float(loss.detach()),float(self.policy.logZ.detach()),beta,time.time()-t0))
        return hist

    @torch.no_grad()
    def sample_counts(self,n=50000):
        from collections import Counter
        C=Counter()
        for _ in range(n):
            t,_,_,_=self.sample(0.0); C[tree_key(t)]+=1
        return C
