"""Generalised rotated surface code, Z-error model (singles + NN pairs). d=3 reproduces avid_model.py layout."""
import numpy as np, itertools, sys

def build(d):
    idx=lambda r,c: r*d+c
    X=[]
    for r in range(-1,d):
        for c in range(-1,d):
            if (r+c)%2==0: continue                      # X-type faces have r+c odd
            sup=[idx(rr,cc) for rr in (r,r+1) for cc in (c,c+1) if 0<=rr<d and 0<=cc<d]
            bulk = 0<=r<d-1 and 0<=c<d-1
            top = r==-1 and 0<=c<d-1 and c%2==0
            bot = r==d-1 and 0<=c<d-1 and (d-1+c)%2==1
            if (bulk or top or bot) and len(sup)>=2: X.append(tuple(sup))
    edges=[(idx(r,c),idx(r,c+1)) for r in range(d) for c in range(d-1)]+\
          [(idx(r,c),idx(r+1,c)) for r in range(d-1) for c in range(d)]
    logical=tuple(idx(r,0) for r in range(d))
    return X,edges,logical

def mechs(d):
    X,edges,L=build(d); n=d*d
    sigq=[sum(1<<k for k,s in enumerate(X) if q in s) for q in range(n)]
    sig=[sigq[q] for q in range(n)]+[sigq[i]^sigq[j] for i,j in edges]
    ell=[int(q in L) for q in range(n)]+[int(i in L)^int(j in L) for i,j in edges]
    sup=[(q,) for q in range(n)]+list(edges)
    return X,np.array(sig),np.array(ell),sup

if __name__=="__main__":
    for d in (3,5):
        X,sig,ell,sup=mechs(d)
        print(f"d={d}: checks={len(X)} (expect {(d*d-1)//2}), mechanisms={len(sig)}, bins={1<<len(X)}")
        print("  X-checks:",X)
        if d==3: print("  matches avid_model d=3 layout:", X==[(0,1),(1,2,4,5),(3,4,6,7),(7,8)] or sorted(X)==sorted([(0,1),(1,2,4,5),(3,4,6,7),(7,8)]))
