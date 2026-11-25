import numpy as np
from d5_fit import *
M=Model(3); base=base_theta(M); Pb=M.P(base); dec_static=M.decision(Pb)
single_free=np.arange(M.m)<M.n; rng=np.random.default_rng(5)
print("d=3 same code/metric, hot single p=0.075, per-qubit gap (static->oracle):")
for q in range(9):
    th=base.copy(); th[q]=0.075; P=M.P(th); st=M.ler(P,dec_static); orc=M.ler(P,M.decision(P)); print(f"  q{q}: static={st:.4f} oracle={orc:.4f} gap={100*(st-orc)/st:.1f}%")
Q=[q for q in range(9) if (lambda P:(M.ler(P,dec_static)-M.ler(P,M.decision(P)))>2e-4)(M.P(np.where(np.arange(M.m)==q,0.075,base)))]
print("qubits with non-trivial gap:",Q)
for N in (256,1024,4096,16384):
    rf=[];rp=[]
    for q in Q:
        th=base.copy(); th[q]=0.075; P=M.P(th); st=M.ler(P,dec_static); gap=st-M.ler(P,M.decision(P))
        for _ in range(8):
            c=rng.multinomial(N,P/P.sum()).astype(float)
            f,_=fit(M,c,np.ones(M.m,bool),base,base,lam=0.05); p,_=fit(M,c,~single_free,base,base,lam=0.05)
            rf.append((st-M.ler(P,M.decision(M.P(f))))/gap); rp.append((st-M.ler(P,M.decision(M.P(p))))/gap)
    print(f"N={N:6d} full mean={np.mean(rf):5.2f} median={np.median(rf):5.2f} | pair-only mean={np.mean(rp):5.2f}")
