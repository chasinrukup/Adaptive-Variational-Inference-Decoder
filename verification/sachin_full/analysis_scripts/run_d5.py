import numpy as np, sys, csv
from d5_fit import *
M=Model(5); base=base_theta(M); Pb=M.P(base); dec_static=M.decision(Pb)
def truth(kind,q=12,pair=(12,13),ps=0.075,pp=0.085):
    th=base.copy()
    if kind=='single': th[q]=ps
    elif kind=='pair': th[M.n+[i for i,s in enumerate(M.sup[M.n:]) if s==pair][0]]=pp
    return th
single_free=np.arange(M.m)<M.n
def run(kind,N,rng,seed_reps=1):
    th=truth(kind); P=M.P(th); dec_or=M.decision(P)
    out=[]
    for r in range(seed_reps):
        if N is None: counts=P*1e9; 
        else: counts=rng.multinomial(N,P/P.sum()).astype(float)
        full,_=fit(M,counts,np.ones(M.m,bool),base,base)
        pairo,_=fit(M,counts,~single_free,base,base)
        row=dict(kind=kind,N=N or 'inf',
                 static=M.ler(P,dec_static),oracle=M.ler(P,dec_or),
                 full=M.ler(P,M.decision(M.P(full))),pair_only=M.ler(P,M.decision(M.P(pairo))),
                 hot_single_est_full=full[12],hot_single_est_pairo=pairo[12])
        out.append(row)
    return out
if __name__=="__main__":
    rng=np.random.default_rng(7); rows=[]
    for kind in ('single','pair'):
        for N in (None,16384,4096,1024,256):
            reps=1 if N is None else 6
            rr=run(kind,N,rng,reps)
            avg={k:(np.mean([x[k] for x in rr]) if isinstance(rr[0][k],float) else rr[0][k]) for k in rr[0]}
            rows.append(avg)
            print(f"{kind:6s} N={str(avg['N']):>6s} static={avg['static']:.4f} oracle={avg['oracle']:.4f} full-fit={avg['full']:.4f} pair-only={avg['pair_only']:.4f}  q12 est full={avg['hot_single_est_full']:.3f} pair-only={avg['hot_single_est_pairo']:.3f}",flush=True)
    with open('results_d5.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
