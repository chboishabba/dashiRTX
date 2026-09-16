#!/usr/bin/env python3
from pathlib import Path
import importlib.util, json
import numpy as np

BASE = Path(__file__).with_name('rsa260_bidi_preparation_fibre_search.py')
spec = importlib.util.spec_from_file_location('prep', BASE)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

SWAP_SEEDS = [2602027,2602029]
PROJECTION_SEEDS = [0,1,2,3]

def cado_grid_permutation(nh=4,nv=4):
    nz=prep.COLS//(nh*nv); p=np.empty(prep.COLS,dtype=int)
    for x in range(prep.COLS):
        q,k=divmod(x,nz); i,j=divmod(q,nv); p[x]=(j*nh+i)*nz+k
    return p

def one_swap(base,seed):
    rng=np.random.default_rng(seed); A=base.copy()
    for r in range(prep.ROWS):
        supp=np.flatnonzero(base[r]); comp=np.flatnonzero(base[r]==0)
        A[r,rng.permutation(supp)[0]]=0; A[r,rng.permutation(comp)[0]]=1
    return A

def main():
    base=prep.build_A(); perm=cado_grid_permutation(4,4); oldA,oldD=prep.A,prep.MAXD; runs=[]
    try:
        prep.MAXD=80
        for sseed in SWAP_SEEDS:
            prep.A=one_swap(base,sseed); apply=prep.make_apply_B(perm)
            for pseed in PROJECTION_SEEDS:
                runs.append({'swap_seed':sseed,'projection_seed':pseed,
                             **prep.evaluate(apply,prep.BASEX^((pseed+1)<<48),prep.BASEY^((pseed+1)<<40))})
    finally:
        prep.A,prep.MAXD=oldA,oldD
    out={'runs':runs,'summary':{'runs':len(runs),'passed':sum(r['passed'] for r in runs),
         'minimum_generator_degree':min(r['degree'] for r in runs),'maximum_generator_degree':max(r['degree'] for r in runs),
         'all_shifted_rank512':all(r['shifted_rank']==512 for r in runs)},
         'firewalls':{'tested_cross_product_is_universal':False,'synthetic_result_is_production_measurement':False}}
    print(json.dumps(out,indent=2)); return out

if __name__=='__main__': main()
