"""Fix test for the alias trap: fit the 21-parameter model on a mix of unlabeled rounds (syndrome only) and a few labeled
calibration rounds (syndrome + true logical bit). Same prior-regularized MAP and the same exact pair-aware decision as
'MAP + exact pair-aware'; the only change is the likelihood, which uses the 32-state joint P(s,l) for labeled rounds."""
import numpy as np
from scipy.optimize import minimize
from avid_model import *; from avid_decoders import theta_prior

V = np.array([int(SIG[k]) | (int(ELL[k]) << 4) for k in range(M)])               # mechanism vectors in Z2^5
_U = np.arange(32)
PAR32 = np.array([[bin(u & int(V[k])).count("1") & 1 for k in range(M)] for u in _U])    # (32,21)
WHT32 = np.array([[(-1)**(bin(x & u).count("1") & 1) for u in _U] for x in _U], float)    # (32,32)

def p_joint32(theta):
    A = 1.0 - 2.0*theta[None, :]*PAR32
    return (WHT32 @ np.prod(A, axis=1)) / 32.0                                      # index s | (l<<4)

def make_nll(Hl, Hu, lam, theta0):
    """Hl: (16,2) labeled counts, Hu: (16,) unlabeled counts. Counts are normalised to probabilities."""
    N = Hl.sum() + Hu.sum(); wl = np.zeros(32); wl[:16] = Hl[:, 0]/N; wl[16:] = Hl[:, 1]/N; wu = Hu/N
    def f(theta):
        A = 1.0 - 2.0*theta[None, :]*PAR32; Ph = np.prod(A, axis=1)
        P = np.maximum((WHT32 @ Ph)/32.0, 1e-300)
        dPh = (-2.0*PAR32) * (Ph[:, None]/A); dP = (WHT32 @ dPh)/32.0
        Ps = np.maximum(P[:16] + P[16:], 1e-300); dPs = dP[:16] + dP[16:]
        nll = -(wl*np.log(P)).sum() - (wu*np.log(Ps)).sum() + 0.5*lam*((theta - theta0)**2).sum()
        grad = -(wl/P) @ dP - (wu/Ps) @ dPs + lam*(theta - theta0)
        return nll, grad
    return f

def fit_map32(Hl, Hu, lam=0.05, maxiter=300):
    th0 = theta_prior()
    r = minimize(make_nll(Hl, Hu, lam, th0), th0.copy(), jac=True, method="L-BFGS-B",
                 bounds=[(1e-5, 0.45)]*M, options={"maxiter": maxiter})
    return r.x

def split_counts(s, l, n_lab):
    Hl = np.zeros((16, 2)); np.add.at(Hl, (s[:n_lab], l[:n_lab]), 1)
    Hu = np.bincount(s[n_lab:], minlength=16).astype(float)
    return Hl, Hu

# ---------- measurement-aware version (circuit level) ----------
POP4 = np.array([bin(u & 15).count("1") for u in range(32)])      # syndrome-bit weight of each Fourier index (logical bit untouched)
def make_nll_q(Hl, Hu, lam, theta0, q_lab, qe_unl):
    """Labeled one-round experiments: events = syndrome XOR one readout flip (prob q_lab per check).
    Unlabeled mid-layer events: syndrome XOR two readout flips (prob qe_unl per check), logical bit marginalised."""
    N = Hl.sum() + Hu.sum(); wl = np.zeros(32); wl[:16] = Hl[:, 0]/N; wl[16:] = Hl[:, 1]/N; wu = Hu/N
    cl = (1 - 2*q_lab) ** POP4; cu = (1 - 2*qe_unl) ** POP4[:16]
    def f(theta):
        A = 1.0 - 2.0*theta[None, :]*PAR32; Ph = np.prod(A, axis=1)
        P = np.maximum((WHT32 @ (Ph*cl))/32.0, 1e-300)
        Ps = np.maximum((WHT32[:16, :16] @ (Ph[:16]*cu))/16.0, 1e-300)
        dPh = (-2.0*PAR32) * (Ph[:, None]/A)
        dP = (WHT32 @ (dPh*cl[:, None]))/32.0
        dPs = (WHT32[:16, :16] @ (dPh[:16]*cu[:, None]))/16.0
        nll = -(wl*np.log(P)).sum() - (wu*np.log(Ps)).sum() + 0.5*lam*((theta - theta0)**2).sum()
        return nll, -(wl/P) @ dP - (wu/Ps) @ dPs + lam*(theta - theta0)
    return f

def fit_map32_q(Hl, Hu, q_lab, qe_unl, lam=0.05, maxiter=300):
    th0 = theta_prior()
    r = minimize(make_nll_q(Hl, Hu, lam, th0, q_lab, qe_unl), th0.copy(), jac=True, method="L-BFGS-B",
                 bounds=[(1e-5, 0.45)]*M, options={"maxiter": maxiter})
    return r.x
