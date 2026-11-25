"""Estimator variants for d=5 (and d=3). Priors expressed in nats so strength does not scale with N.
V0  gaussian prior toward baseline (current)       V1  smoothed-L1 sparsity on excess over baseline
V2  singles free + one global pair scale           V3  weighted LS on low-order Walsh coefficients (compressed syndrome)"""
import numpy as np
from scipy.optimize import minimize
from d5_fit import Model, base_theta, wht, parity

LO,HI=1e-5,0.45
def _opt(f,x0,bounds):
    r=minimize(f,x0,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=400)); return r.x

def fit_v0(M,c,base,pen=2.5,sd=0.01):
    def f(x):
        nll,g=M.nll_grad(x,c); z=(x-base)/sd
        return nll+pen*(z**2).sum(), g+2*pen*z/sd
    return _opt(f,base.copy(),[(LO,HI)]*M.m)

def fit_v1(M,c,base,mu=3.0,sd=0.01,pen_g=0.25,eps=0.2):
    def f(x):
        nll,g=M.nll_grad(x,c); z=(x-base)/sd; r=np.sqrt(z**2+eps**2)
        return nll+mu*r.sum()+pen_g*(z**2).sum(), g+(mu*z/r+2*pen_g*z)/sd
    return _opt(f,base.copy(),[(LO,HI)]*M.m)

def fit_v2(M,c,base,pen=2.5,sd=0.01):
    n=M.n
    def expand(y):
        th=np.r_[y[:n],base[n:]*y[n]]; return th
    def f(y):
        th=expand(y); nll,g=M.nll_grad(th,c); z=(y[:n]-base[:n])/sd
        gy=np.r_[g[:n],(g[n:]*base[n:]).sum()]
        pz=pen*(z**2).sum()+pen*((y[n]-1)/1.0)**2*25      # scale prior sd 0.2
        gy=gy+np.r_[2*pen*z/sd,2*pen*(y[n]-1)*25]
        return nll+pz,gy
    y=_opt(f,np.r_[base[:n],1.0],[(LO,HI)]*n+[(0.05,20)]); return expand(y)

_LOW={}
def low_idx(M,maxw):
    key=(M.d,maxw)
    if key not in _LOW:
        pc=np.array([bin(int(w)).count('1') for w in M.W]); _LOW[key]=np.where((pc>=1)&(pc<=maxw))[0]
    return _LOW[key]
def fit_v3(M,c,base,maxw=3,pen=2.5,sd=0.01):
    N=c.sum(); idx=low_idx(M,maxw); A=M.A[idx]
    emp=(wht(c)/N)[idx]
    mb=np.prod(1-2*base[None,:]*A,axis=1); var=np.maximum(1-mb**2,1e-3)/N
    def f(x):
        fm=1-2*x[None,:]*A; m=np.prod(fm,axis=1); r=emp-m
        J=np.empty_like(A)
        for k in range(M.m): J[:,k]=-2*A[:,k]*np.prod(np.delete(fm,k,axis=1),axis=1)
        z=(x-base)/sd
        return 0.5*(r**2/var).sum()+pen*(z**2).sum(), -(J*(r/var)[:,None]).sum(0)*(-1)*(-1)*1.0*(-1)+2*pen*z/sd
    # careful gradient: d/dx 0.5*sum r^2/var = sum (r/var)*(-dm/dx) = -sum (r/var)*J
    def f2(x):
        fm=1-2*x[None,:]*A; m=np.prod(fm,axis=1); r=emp-m
        J=np.empty_like(A)
        for k in range(M.m): J[:,k]=-2*A[:,k]*np.prod(np.delete(fm,k,axis=1),axis=1)
        z=(x-base)/sd
        return 0.5*(r**2/var).sum()+pen*(z**2).sum(), -(J*(r/var)[:,None]).sum(0)+2*pen*z/sd
    return _opt(f2,base.copy(),[(LO,HI)]*M.m)

VARIANTS={'V0 gauss(current)':lambda M,c,b:fit_v0(M,c,b),
          'V1 L1 mu=1':lambda M,c,b:fit_v1(M,c,b,mu=1.0),
          'V1 L1 mu=3':lambda M,c,b:fit_v1(M,c,b,mu=3.0),
          'V1 L1 mu=10':lambda M,c,b:fit_v1(M,c,b,mu=10.0),
          'V2 singles+pair scale':lambda M,c,b:fit_v2(M,c,b),
          'V3 Walsh<=2':lambda M,c,b:fit_v3(M,c,b,maxw=2),
          'V3 Walsh<=3':lambda M,c,b:fit_v3(M,c,b,maxw=3)}
