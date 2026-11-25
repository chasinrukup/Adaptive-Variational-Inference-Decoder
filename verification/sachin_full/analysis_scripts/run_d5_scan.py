import numpy as np, csv, time
from d5_fit import *
M=Model(5); base=base_theta(M); Pb=M.P(base); dec_static=M.decision(Pb)
single_free=np.arange(M.m)<M.n
rng=np.random.default_rng(11); rows=[]; t0=time.time()
def evaluate(th,N,reps):
    P=M.P(th); dec_or=M.decision(P)
    st,orc=M.ler(P,dec_static),M.ler(P,dec_or); fl=[];po=[]
    for _ in range(reps):
        counts=P*1e9 if N is None else rng.multinomial(N,P/P.sum()).astype(float)
        full,_=fit(M,counts,np.ones(M.m,bool),base,base); pairo,_=fit(M,counts,~single_free,base,base)
        fl.append(M.ler(P,M.decision(M.P(full)))); po.append(M.ler(P,M.decision(M.P(pairo))))
    return st,orc,np.mean(fl),np.mean(po)
for q in range(M.n):
    th=base.copy(); th[q]=0.075
    for N,reps in ((None,1),(1024,3)):
        st,orc,fl,po=evaluate(th,N,reps)
        gap=st-orc; rec=lambda x:(st-x)/gap if gap>1e-6 else float('nan')
        rows.append(dict(q=q,row=q//5,col=q%5,N=N or 'inf',static=st,oracle=orc,full=fl,pair_only=po,gap=gap,rec_full=rec(fl),rec_pair=rec(po)))
        print(f"q={q:2d} (r{q//5}c{q%5}) N={str(N or 'inf'):>4s} static={st:.4f} oracle={orc:.4f} gap={gap:.4f} ({100*gap/st:4.1f}%) recov full={rec(fl):5.2f} pair-only={rec(po):5.2f}",flush=True)
print("elapsed",time.time()-t0)
with open('results_d5_scan.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
