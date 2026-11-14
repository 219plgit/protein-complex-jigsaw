"""Minimal research prototype: constraint-masked graph-attention GFlowNet, exact-flow distillation.

Synthetic undirected contact graphs, no kinetics, not the full PCJ typed-evidence engine.
Each complete object is an unordered rooted binary merger tree; top-down canonical splits
are the actions. Exact teacher counts/weights all legal trees with memoization.
"""
from __future__ import annotations
import argparse, json, random, time, math
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as tF

torch.set_num_threads(1)

def modular_graph(n: int, rng: random.Random, modular=True):
    # A random connected backbone plus within-module contacts.
    adj = np.zeros((n,n), dtype=np.float32)
    module = np.array([min(2, (3*i)//n) if modular else 0 for i in range(n)])
    order=list(range(n));rng.shuffle(order)
    for a,b in zip(order[:-1],order[1:]): adj[a,b]=adj[b,a]=1
    for i in range(n):
      for j in range(i+1,n):
        p=(0.72 if module[i]==module[j] else 0.08) if modular else 0.34
        if rng.random()<p:adj[i,j]=adj[j,i]=1
    # Scalar pair rewards, fixed for the instance (synthetic, not biological kinetics)
    weights=np.zeros((n,n),dtype=np.float32)
    for i in range(n):
      for j in range(i+1,n):
        if adj[i,j]:
          x=rng.uniform(-1.5,1.5)
          weights[i,j]=weights[j,i]=x
    return adj,weights,module

class ExactMerger:
  def __init__(self,adj,weights,beta=1.):
    self.adj=adj;self.weights=weights;self.n=len(adj);self.beta=beta
    self.neighbors=[sum((1<<j) for j in range(self.n) if adj[i,j]) for i in range(self.n)]
  @lru_cache(None)
  def connected(self,mask):
    found=mask&-mask; todo=found
    while todo:
      q=todo&-todo;todo-=q;i=q.bit_length()-1
      nxt=self.neighbors[i]&mask&~found
      found|=nxt;todo|=nxt
    return found==mask
  @lru_cache(None)
  def splits(self,mask):
    if mask.bit_count()<2: return tuple()
    anchor=mask&-mask
    output=[];sub=(mask-1)&mask
    while sub:
      if sub&anchor:
        other=mask^sub
        if self.connected(sub) and self.connected(other) and self.cut_count(sub,other):
          output.append((sub,other))
      sub=(sub-1)&mask
    return tuple(output)
  def cut_count(self,a,b):
    x=0
    for i in range(self.n):
      if (a>>i)&1:
        for j in range(self.n):
          if (b>>j)&1 and self.adj[i,j]:x+=1
    return x
  def log_potential(self,a,b):
    # non-additive, merge-dependent potential: mean crossing score * log(1+cut_count)
    vals=[]
    for i in range(self.n):
      if (a>>i)&1:
        for j in range(self.n):
          if (b>>j)&1 and self.adj[i,j]:vals.append(float(self.weights[i,j]))
    return self.beta * float(np.mean(vals)) * math.log1p(len(vals))
  @lru_cache(None)
  def logZ(self,mask):
    if mask.bit_count()==1:return 0.
    vals=[self.logZ(a)+self.logZ(b)+self.log_potential(a,b) for a,b in self.splits(mask)]
    if not vals:return -math.inf
    ma=max(vals)
    return ma+math.log(sum(math.exp(v-ma) for v in vals))
  def states_and_teacher(self):
    full=(1<<self.n)-1
    if not np.isfinite(self.logZ(full)):raise ValueError('No admissible assembly tree')
    visited=set();pending=[full];result=[]
    while pending:
      mask=pending.pop()
      if mask in visited:continue
      visited.add(mask)
      spl=self.splits(mask)
      if spl:
        ls=np.asarray([self.logZ(a)+self.logZ(b)+self.log_potential(a,b) for a,b in spl])
        prob=np.exp(ls-self.logZ(mask));prob=prob/prob.sum()
        result.append((mask,list(spl),prob))
        for a,b in spl: pending.extend([a,b])
    return result

class HierarchicalPolicy(nn.Module):
  def __init__(self,d=32,heads=4,layers=2):
    super().__init__()
    self.proj=nn.Linear(3,d)
    self.layers=nn.ModuleList([nn.MultiheadAttention(d,heads,batch_first=True) for _ in range(layers)])
    self.norms=nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
    self.scorer=nn.Sequential(nn.Linear(4*d+5,64),nn.ReLU(),nn.Linear(64,1))
  def forward(self,adj,weights,mask,splits):
    n=adj.shape[0]; device=adj.device
    deg=adj.sum(-1,keepdim=True)/max(1,n-1)
    strength=(adj*weights).sum(-1,keepdim=True)/max(1,n-1)
    active=torch.tensor([(mask>>i)&1 for i in range(n)],device=device,dtype=adj.dtype).unsqueeze(1)
    h=self.proj(torch.cat([deg,strength,active],dim=-1)).unsqueeze(0)
    # topological attention bias: forbidden non-contacts cannot supply direct attention
    allowed=(adj>0) | torch.eye(n,device=device,dtype=torch.bool)
    attnmask=torch.where(allowed,torch.tensor(0.,device=device),torch.tensor(-5.,device=device))
    for layer,norm in zip(self.layers,self.norms):
      out,_=layer(h,h,h,attn_mask=attnmask,need_weights=False)
      h=norm(h+out)
    h=h[0]
    def pooled(bits):
      inds=[i for i in range(n) if (bits>>i)&1]
      return h[inds].mean(0)
    parent=pooled(mask)
    feats=[]
    for a,b in splits:
      ia=torch.tensor([i for i in range(n) if (a>>i)&1],device=device)
      ib=torch.tensor([i for i in range(n) if (b>>i)&1],device=device)
      cross=adj[ia][:,ib]; xw=weights[ia][:,ib]
      nz=cross.sum().clamp(min=1.)
      score=(cross*xw).sum()/nz
      s=torch.stack([torch.tensor(a.bit_count()/n,device=device),torch.tensor(b.bit_count()/n,device=device),nz/n,score,torch.abs(score)])
      z1,z2=pooled(a),pooled(b)
      feats.append(torch.cat([parent,z1+z2,torch.abs(z1-z2),z1*z2,s]))
    return self.scorer(torch.stack(feats)).squeeze(-1)

def loss_case(model,case,device):
  adj,wt,examples=case
  a=torch.tensor(adj,device=device); w=torch.tensor(wt,device=device)
  sample=random.sample(examples,min(3,len(examples)))
  loss=[]
  for mask,splits,prob in sample:
    logits=model(a,w,mask,splits)
    tar=torch.tensor(prob,device=device,dtype=torch.float32)
    loss.append(-(tar*tF.log_softmax(logits,dim=0)).sum())
  return torch.stack(loss).mean()

def evaluate(model,cases,device):
  vals=[]; uniform=[]
  with torch.no_grad():
   for adj,wt,examples in cases:
    a=torch.tensor(adj,device=device);w=torch.tensor(wt,device=device)
    for mask,splits,prob in examples:
      if len(splits)<2:continue
      pred=tF.softmax(model(a,w,mask,splits),-1).cpu().numpy()
      vals.append(0.5*np.abs(pred-prob).sum())
      uniform.append(0.5*np.abs(1./len(prob)-prob).sum())
  return dict(mean_local_TV=float(np.mean(vals)),uniform_mean_local_TV=float(np.mean(uniform)),states=len(vals))

def make_cases(count,nlo,nhi,seed):
  cases=[];rng=random.Random(seed)
  for _ in range(count):
    n=rng.randint(nlo,nhi);adj,wt,_=modular_graph(n,rng,modular=rng.random()<0.8)
    ex=ExactMerger(adj,wt)
    data=ex.states_and_teacher()
    cases.append((adj,wt,data))
  return cases

def main():
  p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=12);p.add_argument('--train-graphs',type=int,default=16);p.add_argument('--seed',type=int,default=19);p.add_argument('--output',default='results.json')
  args=p.parse_args();random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
  device='cpu';train=make_cases(args.train_graphs,6,8,args.seed);test=make_cases(4,9,10,args.seed+100)
  model=HierarchicalPolicy().to(device);opt=torch.optim.Adam(model.parameters(),lr=0.001)
  before=evaluate(model,test,device)
  t=time.time()
  for epoch in range(args.epochs):
    random.shuffle(train)
    for c in train:
      opt.zero_grad();ls=loss_case(model,c,device);ls.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
  after=evaluate(model,test,device)
  report={'status':'exploratory synthetic prototype (NOT journal validation)','seed':args.seed,'train_graphs':len(train),'train_n':[6,8],'held_out_graphs':len(test),'test_n':[9,10],'epochs':args.epochs,'before':before,'after':after,'seconds':round(time.time()-t,2),'note':'Local branching TV, not full path TV; held-out graphs synthetic; no biological assembly-time ground truth.'}
  Path(args.output).write_text(json.dumps(report,indent=2));torch.save(model.state_dict(),str(Path(args.output).with_suffix('.pt')))
  print(json.dumps(report,indent=2))
if __name__=='__main__':main()
