"""Generate *candidate* PDB 7ARC inter-chain residue contacts; not assembly order evidence.

Uses Biopython MMCIFParser + scipy.spatial.cKDTree. For each chain pair, records
residues having at least one heavy atom within distance cutoff. Uncertainty:
contacts in a final structure do not establish kinetics, order or even interface
admissibility without review.
"""
import argparse,csv
from pathlib import Path
from Bio.PDB import MMCIFParser
from scipy.spatial import cKDTree
import numpy as np

def chain_coords(chain):
    xyz=[];res=[]
    for residue in chain:
      if residue.id[0]!=' ':continue
      for atom in residue:
        if atom.element.upper() in ('H','D'):continue
        xyz.append(atom.coord);res.append((residue.id[1],residue.id[2]))
    return np.asarray(xyz),res

def main():
    p=argparse.ArgumentParser()
    p.add_argument('cif');p.add_argument('--cutoff',type=float,default=5.)
    p.add_argument('--min-residues',type=int,default=3)
    p.add_argument('--output',default='7arc_candidate_contacts.csv')
    args=p.parse_args()
    assert Path(args.cif).exists(), 'Local mmCIF missing: download PDB 7ARC from RCSB first.'
    structure=MMCIFParser(QUIET=True).get_structure('7ARC',args.cif)
    model=next(iter(structure));chains=[]
    for c in model:
      xyz,res=chain_coords(c)
      if len(xyz)>0:chains.append((c.id,xyz,res))
    rows=[]
    for i,(name_a,xa,ra) in enumerate(chains):
      for name_b,xb,rb in chains[i+1:]:
        pairs=cKDTree(xa).query_ball_tree(cKDTree(xb),args.cutoff)
        r_a=set();r_b=set();num_pairs=0
        for ia, js in enumerate(pairs):
          if js:
            r_a.add(ra[ia]);num_pairs+=len(js)
            for j in js:r_b.add(rb[j])
        if len(r_a)>=args.min_residues and len(r_b)>=args.min_residues:
          rows.append({'chain_a':name_a,'chain_b':name_b,'residues_a':len(r_a),'residues_b':len(r_b),'atom_pairs':num_pairs,'cutoff_A':args.cutoff,'provenance':'PDB 7ARC candidate structural interface; not automatically admitted'})
    with open(args.output,'w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=['chain_a','chain_b','residues_a','residues_b','atom_pairs','cutoff_A','provenance']);w.writeheader();w.writerows(rows)
    print(f'Chains with resolved heavy atoms: {len(chains)}; candidate interfaces: {len(rows)}; output: {args.output}')
if __name__=='__main__':main()
