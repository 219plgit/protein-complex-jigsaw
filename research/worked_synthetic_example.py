"""Fully specified 6-protein synthetic two-module example with one contact deletion.
Exact tree probabilities computed under the SAME non-additive teacher as prototype.py.
"""
from pathlib import Path
from functools import lru_cache
import csv,json,math
import numpy as np
from hgtflow.prototype import ExactMerger
O=Path(__file__).resolve().parent/'benchmark_results'
NAMES=list('ABCDEF'); EDGES={'AB':0.8,'AC':0.5,'BC':0.9,'CD':-0.4,'DE':0.7,'DF':0.45,'EF':1.0}
def mk(remove=None):
 adj=np.zeros((6,6),dtype=np.float32);w=np.zeros_like(adj)
 for e,x in EDGES.items():
  if e==remove:continue
  i,j=map(NAMES.index,e);adj[i,j]=adj[j,i]=1;w[i,j]=w[j,i]=x
 return adj,w

def enumerate_trees(x):
 @lru_cache(None)
 def rec(mask):
  if mask.bit_count()==1:return (((),0.,frozenset([mask])),)
  out=[]
  for a,b in x.splits(mask):
   for la,sa,suba in rec(a):
    for lb,sb,subb in rec(b):
     out.append((((a,b),la,lb),sa+sb+x.log_potential(a,b),suba|subb|frozenset([mask])))
  return tuple(out)
 return rec((1<<6)-1)

def readable(mask):return ''.join(NAMES[i] for i in range(6) if mask>>i&1)
rows=[];summary={}
for condition,removed in [('intact',None),('AC_contact_deleted','AC'),('EF_contact_deleted','EF')]:
 adj,w=mk(removed);ex=ExactMerger(adj,w);trees=enumerate_trees(ex)
 arr=np.array([score for _,score,_ in trees]);p=np.exp(arr-arr.max());p=p/p.sum()
 supports={}
 for subset in ['AB','ABC','DE','DEF','CD','EF']:
  mask=sum(1<<NAMES.index(i) for i in subset)
  supports[subset]=float(sum(pi for pi,(_,_,subs) in zip(p,trees) if mask in subs))
 # most probable top-level split
 roots={}
 for pi,(tree,score,subs) in zip(p,trees):
  a,b=tree[0];root=f'{readable(a)} | {readable(b)}'
  roots[root]=roots.get(root,0)+pi
 top_roots=sorted(roots.items(),key=lambda x:-x[1])[:5]
 summary[condition]={'contacts':int(adj.sum()/2),'admissible_complete_merger_trees':len(trees),'log_partition':float(ex.logZ(63)),'Shannon_entropy_bits':float(-np.sum(p*np.log2(p))), 'effective_number_of_trees':float(2**(-np.sum(p*np.log2(p)))),'subcomplex_support':supports,'top_root_splits':top_roots}
 for key,support in supports.items():rows.append({'condition':condition,'subcomplex':key,'probability':support})
O.mkdir(exist_ok=True)
(O/'worked_example.json').write_text(json.dumps({'proteins':NAMES,'interface_scores':EDGES,'reward':'beta=1, exp(mean interface weight across merge x log(1 + number of crossing interfaces)); product over merges','cases':summary},indent=2))
with (O/'worked_example_support.csv').open('w',newline='') as f:
 ww=csv.DictWriter(f,fieldnames=list(rows[0]));ww.writeheader();ww.writerows(rows)
print(json.dumps(summary,indent=2))
