import numpy as np, collections
from d5_structure import mechs

def parity(x):
    x=np.asarray(x,dtype=np.uint32); p=np.zeros_like(x)
    while x.any(): p^=x&1; x=x>>1
    return p&1

def jac_rank(sig,ell,nbits,joint,theta):
    m=len(sig); W=np.arange(1<<(nbits+(1 if joint else 0)),dtype=np.uint32)
    code=sig.astype(np.uint32)|((ell.astype(np.uint32)<<nbits) if joint else 0)
    a=np.array([parity(W&c) for c in code]).T.astype(float)   # (bins,m) in {0,1}: character is -1 iff a=1
    f=1-2*theta[None,:]*a
    J=np.empty_like(a)
    for k in range(m):
        J[:,k]=-2*a[:,k]*np.prod(np.delete(f,k,axis=1),axis=1)
    s=np.linalg.svd(J,compute_uv=False)
    return int((s>1e-9*s[0]).sum()), s

rng=np.random.default_rng(1)
for d in (3,5):
    X,sig,ell,sup=mechs(d); m=len(sig); nb=len(X)
    theta=rng.uniform(0.005,0.03,m)
    r16,_=jac_rank(sig,ell,nb,False,theta); r32,_=jac_rank(sig,ell,nb,True,theta)
    # distinct signatures, distinct (sig,ell)
    ds=len(set(sig.tolist())); dsl=len(set(zip(sig.tolist(),ell.tolist())))
    print(f"d={d}: mechanisms={m}  rank(syndrome hist)={r16}  rank(joint s,l)={r32}  distinct sig={ds}  distinct (sig,l)={dsl}")
    for name,keyf in (("syndrome only",lambda k:(int(sig[k]),)),("joint (s,l)",lambda k:(int(sig[k]),int(ell[k])))):
        grp=collections.defaultdict(list)
        for k in range(m): grp[keyf(k)].append(sup[k])
        coll=[v for v in grp.values() if len(v)>1]
        sp=[v for v in coll if any(len(x)==1 for x in v) and any(len(x)==2 for x in v)]
        al=sorted({x[0] for v in sp for x in v if len(x)==1})
        print(f"   [{name}] collision groups={len(coll)}, single-vs-pair groups={len(sp)}, singles with a pair alias={len(al)} of {d*d}")
        if name=="syndrome only":
            for v in sp[:6]: print("      e.g.",v)
