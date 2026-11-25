import numpy as np, subprocess
from avid_model import *; from avid_decoders import *; from avid_stim import *
HARN="/tmp/claude-0/-home-claude/3444b933-a437-5e87-b08c-a427ed2d6d1b/scratchpad/cppcheck/harness"
PRIOR=theta_prior()
def mk(**kw):
    t=PRIOR.copy()
    for k,v in kw.items(): t[eval(k)]=v
    return t
SC={"baseline":PRIOR.copy(),"R1 pair":mk(**{"idx_single(4)":.055,"idx_edge(4,5)":.075}),"R3 single":mk(**{"idx_single(4)":.055,"idx_single(3)":.075}),"strong pair":mk(**{"idx_single(4)":.09,"idx_edge(4,5)":.12})}
W=4096; NS=40; rows=[]; hists=[]; keys=[]
for name,th in SC.items():
    sampler=build_one_round(th).compile_detector_sampler(seed=99)
    for s in range(NS):
        det,_=sampler.sample(W,separate_observables=True)
        h=np.bincount((det.astype(np.uint8)<<np.arange(4)).sum(1).astype(int),minlength=16)
        hists.append(h); keys.append(name)
inp="".join("1 "+" ".join(str(int(x)) for x in h)+"\n" for h in hists)
out=subprocess.run([HARN],input=inp,capture_output=True,text=True).stdout.strip().split("\n")
res={n:{"iters":[],"same_dec_conv":[],"ler_py":[],"ler_cpp":[[],[],[]],"dth":[]} for n in SC}
for h,name,line in zip(hists,keys,out):
    th=SC[name]; Pt=dp_joint(th)
    parts=[p.split() for p in line[2:].split("|")[1:]]
    thpy=fit_map(h.astype(float)); tab_py=minweight_table(projected_llr_q78(thpy,PRIOR))
    res[name]["ler_py"].append(ler_of(Pt,tab_py))
    for j,p in enumerate(parts):
        f=float(p[0]); thc=np.array(list(map(float,p[1:22]))); llr=np.array(list(map(int,p[22:31])))
        tab_c=minweight_table(llr); res[name]["ler_cpp"][j].append(ler_of(Pt,tab_c))
        if j==2:
            res[name]["same_dec_conv"].append(bool((tab_c==tab_py).all())); res[name]["dth"].append(np.abs(effective_p(thc)-effective_p(thpy)).max())
print(f"{'scenario':12s} {'LER python (conv.)':>18s} {'C++ 20 it':>10s} {'C++ 22 it':>10s} {'C++ 300 it':>11s} {'same table (C++300 vs py)':>26s} {'max|d p_eff|':>13s}")
for n,r in res.items():
    print(f"{n:12s} {100*np.mean(r['ler_py']):17.2f}% {100*np.mean(r['ler_cpp'][0]):9.2f}% {100*np.mean(r['ler_cpp'][1]):9.2f}% {100*np.mean(r['ler_cpp'][2]):10.2f}% {np.mean(r['same_dec_conv']):26.2f} {np.max(r['dth']):13.4f}")
