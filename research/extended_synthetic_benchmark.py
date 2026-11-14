"""Exploratory exact-teacher vs frozen PCJ-HGT prototype benchmark.
No trajectory-balance training, no biological data, no external baseline claims.
"""
import argparse,csv,json,math,random,time,sys
from pathlib import Path
import numpy as np, torch
from scipy.stats import bootstrap
from hgtflow.prototype import ExactMerger,HierarchicalPolicy,modular_graph

torch.set_num_threads(1)

def entropy(p):
 p=np.asarray(p,dtype=float); p=p[p>0]; return float(-(p*np.log2(p)).sum())
def metrics(p,q):
 p=np.asarray(p,dtype=float);q=np.asarray(q,dtype=float);eps=1e-12;q=np.maximum(q,eps);q=q/q.sum()
 return {'tv':float(np.abs(p-q).sum()/2),'kl':float(np.sum(p*np.log(np.maximum(p,eps)/q))), 'top1':float(np.argmax(p)==np.argmax(q)), 'neff_error':float(abs(2**entropy(p)-2**entropy(q))), 'entropy_error':float(abs(entropy(p)-entropy(q)))}

def run(out,graphs_per_size=2,n_sizes=(6,8,9),teacher_seed=20261114):
 src=Path(__file__).resolve().parent;models=[]
 for tag in ['','_seed20','_seed21']:
  f=src/'hgtflow'/('prototype_results'+tag+'.pt')
  if not f.exists():continue
  m=HierarchicalPolicy();m.load_state_dict(torch.load(f,map_location='cpu',weights_only=True));m.eval();models.append((tag or '_seed19',m))
 rows=[]; start=time.time();rng=random.Random(teacher_seed)
 for n in n_sizes:
  for gi in range(graphs_per_size):
   modular=(gi%2==0);adj,w,_=modular_graph(n,rng,modular=modular)
   teacher=ExactMerger(adj,w)
   examples=teacher.states_and_teacher()
   ta=torch.tensor(adj,dtype=torch.float32);tw=torch.tensor(w,dtype=torch.float32)
   for mask,splits,p in examples:
    if len(splits)<2:continue
    cut_scores=np.asarray([teacher.log_potential(a,b) for a,b in splits]);cut_scores-=cut_scores.max();heur=np.exp(cut_scores);heur=heur/heur.sum()
    predictions=[('uniform',np.ones(len(p))/len(p)),('local_potential',heur)]
    with torch.no_grad():
     for name,model in models:
      logits=model(ta,tw,mask,splits).numpy();logits-=logits.max();q=np.exp(logits);predictions.append(('HGT'+name,q/q.sum()))
    for model,q in predictions:
     data=metrics(p,q)
     rows.append({'n':n,'graph':f'{n}_{gi}','topology':'modular' if modular else 'nonmodular','state_size':mask.bit_count(),'splits':len(splits),'model':model,**data})
  print('Completed n=',n,'elapsed',round(time.time()-start,2),flush=True)
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 with (out/'state_metrics.csv').open('w',newline='') as f:
  ww=csv.DictWriter(f,fieldnames=list(rows[0]));ww.writeheader();ww.writerows(rows)
 # State-weighted means are descriptive; graph-averaged results prevent large graphs dominating.
 summary=[]
 for n in n_sizes:
  for model in sorted(set(x['model'] for x in rows)):
   g_rows=[]
   for gi in range(graphs_per_size):
    rr=[x for x in rows if x['n']==n and x['model']==model and x['graph']==f'{n}_{gi}']
    if rr:g_rows.append({k:float(np.mean([x[k] for x in rr])) for k in ['tv','kl','top1','entropy_error','neff_error']})
   if not g_rows:continue
   summary.append({'n':n,'model':model,'graphs':len(g_rows),'states':len([x for x in rows if x['n']==n and x['model']==model]),**{k:float(np.mean([x[k] for x in g_rows])) for k in ['tv','kl','top1','entropy_error','neff_error']},'tv_graph_sd':float(np.std([x['tv'] for x in g_rows],ddof=1)) if len(g_rows)>1 else None})
 with (out/'summary.csv').open('w',newline='') as f:
  ww=csv.DictWriter(f,fieldnames=list(summary[0]));ww.writeheader();ww.writerows(summary)
 config={'seed':teacher_seed,'graph_sizes':list(n_sizes),'graphs_per_size':graphs_per_size,'frozen_models':[s for s,_ in models], 'beta':1.,'scope':'Exploratory teacher-distillation sanity check, NOT trajectory-balance GFlowNet validation.','evaluation':'All nontrivial reachable subset states; macro average by graph; 2 graphs/size insufficient inferential claims','time_s':round(time.time()-start,2),'limitations':['Synthetic contact graphs without protein-specific prerequisites or exclusions','Only local conditional split distributions, not full trajectory distribution','No held-out experimentally measured intermediates','Graphs share synthetic generation regime with original training','Model trained at 6-8 chains; 9 chains extrapolates','Small number of graphs; no confidence interval claimed']}
 (out/'metadata.json').write_text(json.dumps(config,indent=2))
 print(json.dumps(summary,indent=2))
 return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',default='synthetic_benchmark');p.add_argument('--graphs-per-size',type=int,default=2);args=p.parse_args();run(args.output,args.graphs_per_size)
