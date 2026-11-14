import random
import numpy as np
from prototype import ExactMerger, modular_graph, HierarchicalPolicy
import torch

def test_canonical_trees():
  # Complete four-vertex graph: (2n-3)!! = 15 unordered rooted binary trees.
  n=4;adj=np.ones((n,n),dtype=np.float32)-np.eye(n,dtype=np.float32)
  m=ExactMerger(adj,np.zeros((n,n),dtype=np.float32))
  assert round(np.exp(m.logZ(15)))==15
  for mask,splits,prob in m.states_and_teacher():
    assert np.isclose(prob.sum(),1.,atol=1e-8)
    for a,b in splits:
      assert a&b==0 and a|b==mask and a&-a==mask&-mask

def test_policy():
  adj,wt,_=modular_graph(7,random.Random(2))
  ex=ExactMerger(adj,wt)
  mask,splits,prob=ex.states_and_teacher()[0]
  m=HierarchicalPolicy()
  logits=m(torch.tensor(adj),torch.tensor(wt),mask,splits)
  assert logits.numel()==len(splits)
  assert torch.isfinite(logits).all()

if __name__=='__main__':
 test_canonical_trees(); test_policy();print('two prototype tests passed')
