"""d=5 estimator test: hot single vs hot pair, full-model fit vs pair-only (repo-style) fit, exact and finite-sample."""
import numpy as np, sys
from scipy.optimize import minimize
from d5_structure import mechs

def wht(a):
    a=a.copy(); h=1; n=len(a)
    while h<n:
        a=a.reshape(-1,2,h); a=np.stack([a[:,0]+a[:,1],a[:,0]-a[:,1]],1).reshape(-1); h*=2
    return a
def parity(x):
    x=np.asarray(x,dtype=np.uint32); p=np.zeros_like(x)
    while x.any(): p^=x&1; x=x>>1
    return p&1

class Model:
    def __init__(s,d):
        s.d=d; s.X,s.sig,s.ell,s.sup=mechs(d); s.nb=len(s.X); s.m=len(s.sig); s.n=d*d
        s.code=(s.sig.astype(np.uint32)|(s.ell.astype(np.uint32)<<s.nb))
        s.W=np.arange(1<<(s.nb+1),dtype=np.uint32)
        s.A=np.array([parity(s.W&c) for c in s.code]).T.astype(float)   # (bins,m)
    def P(s,theta):                      # exact joint P(s,l) over 2^(nb+1) bins, index = s | l<<nb
        f=1-2*theta[None,:]*s.A
        what=np.prod(f,axis=1)
        return np.clip(wht(what)/len(what),1e-300,None)
    def nll_grad(s,theta,counts):
        f=1-2*theta[None,:]*s.A; what=np.prod(f,axis=1); P=np.clip(wht(what)/len(what),1e-300,None)
        nll=-(counts*np.log(P)).sum()
        r=wht(counts/P)/len(what)       # dNLL/dwhat = -wht(c/P)/N  (wht symmetric)
        dw=np.empty_like(s.A)
        for k in range(s.m):
            dw[:,k]=-2*s.A[:,k]*np.prod(np.delete(f,k,axis=1),axis=1)
        return nll, -(r[:,None]*dw).sum(0)
    def decision(s,P):
        nbins=1<<s.nb; P0=P[:nbins]; P1=P[nbins:]
        return (P1>P0).astype(int)
    def ler(s,P,dec):
        nbins=1<<s.nb; return float(np.where(dec==1,P[:nbins],P[nbins:]).sum())

def base_theta(M,single=0.012,pair=0.004):
    return np.r_[np.full(M.n,single),np.full(M.m-M.n,pair)]

def fit(M,counts,mask,theta0,prior,lam=0.05,sprior=None):
    """MAP over free params (mask True); others fixed at theta0. Gaussian-ish prior toward `prior`."""
    free=np.where(mask)[0]
    # simple scaled objective: mean NLL + lam/ N_eff * penalty
    N=counts.sum()
    def f(x):
        th=theta0.copy(); th[free]=x
        nll,g=M.nll_grad(th,counts)
        pen=0.5*lam*(((x-prior[free])/0.01)**2).sum()
        return nll/N+pen/ max(N,1)*100, (g[free]/N+lam*(x-prior[free])/0.01**2/ max(N,1)*100)
    res=minimize(f,theta0[free],jac=True,method='L-BFGS-B',bounds=[(1e-5,0.45)]*len(free),options=dict(maxiter=500))
    th=theta0.copy(); th[free]=res.x; return th,res
