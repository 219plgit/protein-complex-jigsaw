from __future__ import annotations
from functools import lru_cache
import random

class ExactMergerFlow:
    """Exact reward-proportional flow over canonical unordered merger trees.

    merge_weight(A,B,S) is a non-negative local potential. With unit weight,
    flow(S) is the exact number of admissible merger trees over S.
    """
    def __init__(self, model, merge_weight=None):
        self.model=model
        self.merge_weight=merge_weight or (lambda A,B,S:1.0)

    @lru_cache(None)
    def flow(self,S=None):
        if S is None: S=self.model.full
        if S & (S-1)==0: return 1.0
        total=0.0
        for A,B in self.model.splits(S):
            w=float(self.merge_weight(A,B,S))
            if w < 0: raise ValueError("merge weights must be non-negative")
            total += w*self.flow(A)*self.flow(B)
        return total

    def root_splits(self,S=None):
        if S is None: S=self.model.full
        z=self.flow(S)
        out=[]
        for A,B in self.model.splits(S):
            w=float(self.merge_weight(A,B,S)); mass=w*self.flow(A)*self.flow(B)
            if mass>0: out.append((A,B,mass/z))
        return out

    def sample_tree(self,S=None,rng=None):
        if S is None: S=self.model.full
        rng=rng or random.Random()
        if S & (S-1)==0: return S
        rows=self.root_splits(S)
        if not rows: raise ValueError("no admissible completion")
        x=rng.random(); acc=0.0; A=B=None
        for a,b,p in rows:
            acc += p
            if x<=acc: A,B=a,b; break
        if A is None: A,B,_=rows[-1]
        return (S,self.sample_tree(A,rng),self.sample_tree(B,rng))
