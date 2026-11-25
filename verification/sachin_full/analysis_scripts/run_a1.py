import numpy as np, pickle, time
from avid_model import *; from avid_decoders import *
PRIOR = theta_prior()
burst = PRIOR.copy(); burst[idx_single(4)] = 0.055; burst[idx_edge(4,5)] = 0.075     # R1: repo correlated hotspot
Pm_base, Pm_burst = p_syndrome(PRIOR), p_syndrome(burst)
rng = np.random.default_rng(2026)
WS = [16, 32, 64, 128, 256, 512, 1024, 4096, 10000]
FG = np.linspace(0, 1, 21); NDRAW = 200
tables = {}; probs = {}            # probs[W] -> (len(FG), n_tables) filled later
t0 = time.time()
rec = {}
for W in WS:
    rec[W] = []
    for f in FG:
        nb = int(round(f*W)); ids = []
        for _ in range(NDRAW):
            h = rng.multinomial(nb, Pm_burst) + rng.multinomial(W-nb, Pm_base)
            th = fit_map(h.astype(float))
            tab = tuple(int(x) for x in minweight_table(projected_llr_q78(th, PRIOR)))
            ids.append(tables.setdefault(tab, len(tables)))
        rec[W].append(np.bincount(ids, minlength=64))
    print(f"[{time.time()-t0:4.0f}s] W={W} done, distinct tables so far {len(tables)}", flush=True)
n = len(tables)
probs = {W: np.array([r[:n] if len(r)>=n else np.pad(r,(0,n-len(r))) for r in rec[W]], float)/NDRAW for W in WS}
pickle.dump({"tables":tables,"probs":probs,"FG":FG,"WS":WS}, open("a1.pkl","wb"))
# summary: which tables, and P(table == burst-oracle-like) vs f
Pj_burst, Pj_base = dp_joint(burst), dp_joint(PRIOR)
inv = {v:k for k,v in tables.items()}
print("\ndistinct tables:", n)
for i in range(n):
    t = np.array(inv[i]); print(i, "LER under burst %.2f%%  under baseline %.2f%%"%(100*ler_of(Pj_burst,t),100*ler_of(Pj_base,t)), "bits@0x4=",t[4])
print("oracle under burst %.2f%%, static under burst %.2f%%"%(100*ler_opt(Pj_burst), 100*ler_of(Pj_burst, minweight_table(marginal_llr_q78(PRIOR)))))
print("\nP(good decision) = P(table with burst LER<5%) by burst fraction f, per window W")
good = [i for i in range(n) if ler_of(Pj_burst,np.array(inv[i]))<0.05]
print("f      "+"  ".join(f"{f:4.2f}" for f in FG[::2]))
for W in WS: print(f"W={W:5d} "+"  ".join(f"{probs[W][j,good].sum():4.2f}" for j in range(0,21,2)))
