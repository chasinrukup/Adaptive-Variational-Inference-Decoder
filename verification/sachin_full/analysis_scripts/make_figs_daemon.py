import pandas as pd, numpy as np, json, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
d=pd.read_csv('../daemon_runs.csv')
ctrls=["static","marginal_lbfgs","joint_dp_proj","avid_r2d2","b4_oracle"]
lab={"static":"static","marginal_lbfgs":"marginal L-BFGS","joint_dp_proj":"joint DP + projection","avid_r2d2":"AVID + trained RL","b4_oracle":"oracle (true rates)"}
col={"static":"#5B7C99","marginal_lbfgs":"#8AA4BD","joint_dp_proj":"#F2A33A","avid_r2d2":"#0E9AA7","b4_oracle":"#2E7D5B"}
scs=[("correlated_noise","pair hot (repo scenario)"),("alias_single_hot","single qubit hot (added)")]
fig,ax=plt.subplots(1,2,figsize=(11.5,4.0),gridspec_kw={"width_ratios":[1.15,1]})
w=0.16
for i,(sc,t) in enumerate(scs):
    for j,c in enumerate(ctrls):
        v=100*d[(d.scenario==sc)&(d.controller==c)].ler; m=v.mean(); e=1.96*v.std()/np.sqrt(len(v))
        ax[0].bar(i+(j-2)*w,m,w*0.92,yerr=e,color=col[c],label=lab[c] if i==0 else None,capsize=2)
        ax[0].text(i+(j-2)*w,m+e+0.15,f"{m:.1f}",ha="center",fontsize=8)
ax[0].set_xticks([0,1]); ax[0].set_xticklabels([t for _,t in scs]); ax[0].set_ylabel("logical error rate (%)"); ax[0].set_ylim(0,14)
ax[0].set_title("Repository daemon, 10 seeds (95% intervals)",fontsize=10); ax[0].legend(fontsize=8,frameon=False,ncol=3,loc="upper center")
R=json.load(open('../daemon_rl_full.json'))
names=["STATIC_PRIOR_K3 (pair scale 0.00)","ADAPTIVE_PAIR_K4 (1.15)","AGGRESSIVE_PAIR_K5 (1.35)","DAMPED_PRIOR_K4 (1.05)","FAST_RECALIBRATE_K5 (1.25)","CONSERVATIVE_FALLBACK_K3 (0.00)"]
cnt={s:np.zeros(6) for s,_ in scs}
for r in R:
    if r['scenario'] in cnt: cnt[r['scenario']]+=np.array(r['r2d2_action_histogram'])
x=np.arange(6)[::-1]
ax[1].barh(x+0.2,cnt["correlated_noise"],0.38,color="#0E9AA7",label="pair hot"); ax[1].barh(x-0.2,cnt["alias_single_hot"],0.38,color="#C8433B",label="single hot")
ax[1].set_yticks(x); ax[1].set_yticklabels([n.replace("\n"," ") for n in names],fontsize=7.5); ax[1].set_xlabel("decisions (all seeds)")
for xi,(a1,a2) in zip(x,zip(cnt["correlated_noise"],cnt["alias_single_hot"])):
    ax[1].text(a1+2,xi+0.2,f"{int(a1)}",va="center",fontsize=8); ax[1].text(a2+2,xi-0.2,f"{int(a2)}",va="center",fontsize=8)
ax[1].set_xlim(0,165); ax[1].set_title("Trained RL policy: action counts (346 decisions)",fontsize=10); ax[1].legend(fontsize=8,frameon=False,loc="lower right")
plt.tight_layout(); plt.savefig("fig_daemon.png",dpi=200)
print({k:v.tolist() for k,v in cnt.items()})
