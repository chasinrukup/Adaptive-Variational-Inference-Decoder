import numpy as np, csv, sys, time
from d5_variants import *
M=Model(5); base=base_theta(M); dec_static=M.decision(M.P(base)); rng=np.random.default_rng(101)
def truth(kind):
    th=base.copy()
    if kind[0]=='s': th[kind[1]]=0.075
    else: th[M.n+[i for i,s in enumerate(M.sup[M.n:]) if s==kind[1]][0]]=0.085
    return th
TEST=[('s',q) for q in (10,14,19,6,18,15,20,24)]+[('p',(12,13)),('p',(10,15)),('p',(7,12))]
N_LIST=(1024,4096); REPS=6; t0=time.time(); rows=[]
for name,fn in VARIANTS.items():
    for N in N_LIST:
        rec={'s':[],'p':[]}; gains={'s':[0.0,0.0],'p':[0.0,0.0]}
        for kind in TEST:
            th=truth(kind); P=M.P(th); st=M.ler(P,dec_static); gap=st-M.ler(P,M.decision(P))
            for _ in range(REPS):
                c=rng.multinomial(N,P/P.sum()).astype(float)
                est=fn(M,c,base); g=st-M.ler(P,M.decision(M.P(est))); rec[kind[0]].append(g/gap); gains[kind[0]][0]+=g; gains[kind[0]][1]+=gap
        r=dict(variant=name,N=N,single_pooled=gains['s'][0]/gains['s'][1],pair_pooled=gains['p'][0]/gains['p'][1],single_mean=np.mean(rec['s']),single_med=np.median(rec['s']),single_neg=np.mean(np.array(rec['s'])<0),pair_mean=np.mean(rec['p']),pair_neg=np.mean(np.array(rec['p'])<0))
        rows.append(r); print(f"{name:24s} N={N:5d} hot-single pooled={r['single_pooled']:5.2f} mean={r['single_mean']:5.2f} med={r['single_med']:5.2f} frac<0={r['single_neg']:.2f} | hot-pair pooled={r['pair_pooled']:5.2f} frac<0={r['pair_neg']:.2f}   [{time.time()-t0:.0f}s]",flush=True)
with open('results_d5_variants.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
