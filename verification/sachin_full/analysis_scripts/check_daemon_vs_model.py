import numpy as np, pandas as pd, sys
sys.argv=['x']; import run_sweep as rs
from avid_model import *; from avid_decoders import *
PRIOR=theta_prior(); thb=PRIOR.copy(); thb[idx_single(4)]=0.075; thb[idx_edge(4,5)]=0.085        # daemon: burst_disturbance / short_burst
Pm_b,Pj_b=p_syndrome(thb),dp_joint(thb); SB=ler_of(Pj_b,rs.STATIC); OB=ler_opt(Pj_b)
print("model: static in-burst LER %.2f%%, optimum %.2f%%  (daemon static: 9.46-9.55%%)"%(100*SB,100*OB))
L=rs.draw_losses(thb,Pm_b,Pj_b); rng=np.random.default_rng(5)
def overlap(a0,a1,b0,b1): return np.maximum(0.0,np.minimum(a1,b1)-np.maximum(a0,b0))
def inburst(W,t0,t1,D,H,n=600):
    LB,L0=L["proj"][W]; out=[]
    for _ in range(n):
        ek=np.arange(1,int(H//W)+1)*W; f=overlap(ek-W,ek,t0,t1)/W; fi=np.clip(np.rint(f*(len(rs.FG)-1)).astype(int),0,len(rs.FG)-1)
        j=rng.integers(0,rs.ND,len(ek)); lb=LB[fi,j]; commit=ek+D
        starts=np.concatenate([[0.0],commit]); ends=np.concatenate([commit,[H]]); ids=np.concatenate([[SB],lb])
        ok=starts<H; starts,ends,ids=starts[ok],np.minimum(ends[ok],H),ids[ok]
        out.append((overlap(starts,ends,t0,t1)*ids).sum()/(t1-t0))
    return np.mean(out)
d=pd.read_csv("../daemon_track.csv")
print(f"{'scenario':26s} {'W':>5s} {'D':>5s} | {'model in-burst LER %':>21s} {'reduction vs static':>20s} | {'daemon joint_dp_proj %':>23s} {'reduction':>10s} {'daemon avid_r2d2 %':>19s}")
for sc,(t0,t1) in {"burst_disturbance":(1024,3072),"short_burst_60us_vs_10ms":(196,1280)}.items():
    for W in (256,1024):
        g=d[(d._scenario==sc)&(d._window==W)]
        dj=g[g._controller=="joint_dp_proj"].burst_window_avid_ler.mean(); dr=g[g._controller=="avid_r2d2"].burst_window_avid_ler.mean(); ds=g[g._controller=="static"].burst_window_avid_ler.mean()
        for D in (0,1500):
            m=inburst(W,t0,t1,D,4096)
            dj_s=f"{100*dj:23.2f} {100*(ds-dj)/ds:9.1f}% {100*dr:19.2f}" if D==0 else " "*23+" "*11+" "*19
            print(f"{sc:26s} {W:5d} {D:5d} | {100*m:21.2f} {100*(SB-m)/SB:19.1f}% | {dj_s}")
