#!/usr/bin/env python3
"""Exploratory 7ARC graph-attention analysis. Not a trained trajectory-balance GFlowNet.
Provide --cif 7ARC.cif to produce genuine coordinate-derived candidate contacts.
Without --cif, --synthetic performs a 16-node *synthetic* stress test only.
Capped split shortlist is approximate and cannot be interpreted as calibrated mechanism posterior.
"""
import sys, argparse, json, csv, random, math
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'hgtflow'))
from prototype import ExactMerger, HierarchicalPolicy, modular_graph

def from_cif(path, cutoff, min_residues):
 from Bio.PDB import MMCIFParser
 model=next(iter(MMCIFParser(QUIET=True).get_structure('7ARC',str(path))))
 chains=[]
 for chain in model:
  xyz=[];resids=[]
  for res in chain:
   if res.id[0]!=' ': continue
   for atom in res:
    if atom.element.upper() in ('H','D'):continue
    xyz.append(atom.coord);resids.append((res.id[1],res.id[2]))
  if xyz: chains.append((chain.id,np.array(xyz),resids))
 n=len(chains)
 if n!=16:raise ValueError(f'Expected 16 resolved chains for PDB 7ARC; observed {n}. Check selected structure/assembly.')
 adj=np.zeros((n,n),dtype=np.float32); weights=np.zeros_like(adj); contacts=[]
 for i,(ch_a,xa,ra) in enumerate(chains):
  for j in range(i+1,n):
   ch_b,xb,rb=chains[j]
   pairs=cKDTree(xa).query_ball_tree(cKDTree(xb),cutoff)
   residue_a={ra[k] for k,js in enumerate(pairs) if js}
   residue_b={rb[q] for js in pairs for q in js}
   if len(residue_a)>=min_residues and len(residue_b)>=min_residues:
    num=sum(map(len,pairs))
    adj[i,j]=adj[j,i]=1
    contacts.append({'chain_a':ch_a,'chain_b':ch_b,'residues_a':len(residue_a),'residues_b':len(residue_b),'atom_pairs':num})
 # Artificial bounded scoring scale, NOT experimental energetic preference.
 if contacts:
  vals=np.array([math.log1p(c['atom_pairs']) for c in contacts]); mean=float(vals.mean()); st=float(max(vals.std(),1.))
  for c in contacts:
   i=next(k for k,x in enumerate(chains) if x[0]==c['chain_a']);j=next(k for k,x in enumerate(chains) if x[0]==c['chain_b'])
   weights[i,j]=weights[j,i]=float(np.clip((math.log1p(c['atom_pairs'])-mean)/st,-1.5,1.5))
 return adj,weights,[c[0] for c in chains],contacts

def shortlist(ex,mask,k,rng):
 possibilities=list(ex.splits(mask))
 if len(possibilities)<=k:return possibilities,len(possibilities)
 # Diversity/random balance; shortlist is strictly a computational approximation.
 ranked=sorted(possibilities,key=lambda p:(-ex.cut_count(*p),min(p[0].bit_count(),p[1].bit_count())))
 keep=ranked[:k//2]
 remaining=ranked[k//2:]
 keep+=rng.sample(remaining,min(k-len(keep),len(remaining)))
 return keep,len(possibilities)

def sample_model(adj,weights,labels,model_path,seed,nsamp,max_splits):
 torch.manual_seed(seed);rng=random.Random(seed)
 model=HierarchicalPolicy()
 model.load_state_dict(torch.load(str(model_path),map_location='cpu',weights_only=True));model.eval()
 ex=ExactMerger(adj,weights)
 n=len(adj);root=(1<<n)-1
 if not ex.connected(root):return {'error':'Candidate contact graph disconnected. No fully connected assembly under connectivity-only constraint.'}
 top=Counter(); intermediate=Counter(); lengths=[]; capped=0; max_candidates=0
 a=torch.tensor(adj);w=torch.tensor(weights)
 with torch.no_grad():
  for sample in range(nsamp):
   todo=[root]; internal=0
   while todo:
    mask=todo.pop()
    if mask.bit_count()==1:continue
    choices,all_count=shortlist(ex,mask,max_splits,rng)
    if not choices:raise ValueError('No legal split for reachable connected subset')
    max_candidates=max(max_candidates,all_count)
    if len(choices)<all_count:capped+=1
    logits=model(a,w,mask,choices)
    probs=torch.softmax(logits,dim=0).cpu().numpy()
    pick=int(rng.choices(range(len(choices)),weights=probs,k=1)[0])
    suba,subb=choices[pick]
    if mask==root:top[tuple(sorted((suba.bit_count(),subb.bit_count())))]+=1
    for sub in (suba,subb):
     if 1<sub.bit_count()<n:intermediate[sub]+=1
     if sub.bit_count()>1:todo.append(sub)
    internal+=1
   lengths.append(internal)
 def lab(mask):return [labels[i] for i in range(n) if (mask>>i)&1]
 most=[{'chains':lab(mask),'size':mask.bit_count(),'frequency':round(v/nsamp,3)} for mask,v in intermediate.most_common(12)]
 return {'checkpoint':model_path.name,'samples':nsamp,'top_root_split_sizes':{str(k):v for k,v in top.most_common()},'most_frequent_intermediates':most,'max_legal_splits_at_state':max_candidates,'num_truncated_decisions':capped,'each_tree_internal_nodes':int(np.mean(lengths))}

def main():
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--cif');g.add_argument('--synthetic',action='store_true')
 ap.add_argument('--cutoff',type=float,default=5.);ap.add_argument('--min-residues',type=int,default=3);ap.add_argument('--samples',type=int,default=30);ap.add_argument('--max-splits',type=int,default=128);ap.add_argument('--output',default='7arc_hgt_analysis.json')
 args=ap.parse_args()
 if args.synthetic:
  adj,weights,_=modular_graph(16,random.Random(20261113),True);labels=[f'SYN_{i+1:02d}' for i in range(16)]; contacts=[]
  kind='SYNTHETIC 16-node graph; NOT 7ARC results'
 else:
  adj,weights,labels,contacts=from_cif(Path(args.cif),args.cutoff,args.min_residues);kind='PDB 7ARC coordinate-derived candidate interface graph; NOT verified assembly order'
 checkpoints=sorted((ROOT/'hgtflow').glob('prototype_results*.pt'))
 runs=[sample_model(adj,weights,labels,p,19+i,args.samples,args.max_splits) for i,p in enumerate(checkpoints)]
 report={'analysis_type':kind,'pdb_id':'7ARC' if args.cif else None,'n_chains':len(labels),'chain_names':labels,'n_candidate_edges':int(np.triu(adj,1).sum()),'candidate_contacts':contacts,'policy':'synthetic-trained GNN with hard connectivity masking, approximate capped split proposal','training_scope':'trained on synthetic n=6..8, evaluated previously n=9..10; 16-node extrapolation is out-of-distribution','not_demonstrated':['Trajectory balance','calibrated path probabilities','biological assembly route','true 7ARC assembly kinetics','uncertainty calibration'],'params':vars(args),'runs':runs}
 p=Path(args.output);p.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('candidate_contacts','runs')},indent=2));print('OUTPUT',p)
 if args.cif:
  with p.with_suffix('.contacts.csv').open('w',newline='') as f:
   cw=csv.DictWriter(f,fieldnames=['chain_a','chain_b','residues_a','residues_b','atom_pairs']);cw.writeheader();cw.writerows(contacts)
if __name__=='__main__':main()
