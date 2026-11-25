import numpy as np, csv
import d5_fit
from d5_fit import *
M=Model(5); base=base_theta(M); Pb=M.P(base); dec_static=M.decision(Pb)
single_free=np.arange(M.m)<M.n
rng=np.random.default_rng(29); rows=[]
def pd_write():
    if rows:
        with open('results_d5_lam_b.csv','w',newline='') as fh:
            w=csv.DictWriter(fh,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
Q=[0,5,10,14,19,12,6,18]            # corner, column-0/4 edges, center, inner
for lam in (5.0,):
    for N in (4096,16384,65536):
        recf=[];recp=[]
        for q in Q:
            th=base.copy(); th[q]=0.075; P=M.P(th); dec_or=M.decision(P)
            st=M.ler(P,dec_static); orc=M.ler(P,dec_or); gap=st-orc
            for _ in range(8):
                c=rng.multinomial(N,P/P.sum()).astype(float)
                f,_=fit(M,c,np.ones(M.m,bool),base,base,lam=lam); p,_=fit(M,c,~single_free,base,base,lam=lam)
                recf.append((st-M.ler(P,M.decision(M.P(f))))/gap); recp.append((st-M.ler(P,M.decision(M.P(p))))/gap)
        rows.append(dict(lam=lam,N=N,rec_full_mean=np.mean(recf),rec_full_med=np.median(recf),rec_pair_mean=np.mean(recp)))
        pd_write()
        print(f"lam={lam:5.2f} N={N:6d}  full mean={np.mean(recf):5.2f} median={np.median(recf):5.2f} frac<0={np.mean(np.array(recf)<0):.2f} | pair-only mean={np.mean(recp):5.2f}",flush=True)
with open('results_d5_lam_b.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
