"""Task 4: multi-round circuit-level (phenomenological) extension of the Stim cross-check.
R noisy rounds: Z / ZZ events with probabilities theta before each round, noisy X-check readout (flip prob q),
then one perfect final readout. Detector layer r (0..R) = check outcome of round r+1 XOR round r. The decoder is a
space-time PyMatching graph whose space-edge weights come from the 9 per-qubit LLRs and whose time-edge weight
is ln((1-q)/q)."""
import stim, numpy as np, math, pymatching
from scipy.optimize import minimize
from avid_model import *; from avid_decoders import *; from avid_stim import _checks_str

def build_circuit(theta, q, R):
    c = stim.Circuit(); c.append("RX", list(range(N_Q)))
    for r in range(R):
        for k in range(M):
            qs = SUPPORT[k]
            if len(qs) == 1: c.append("Z_ERROR", [qs[0]], float(theta[k]))
            else: c.append("CORRELATED_ERROR", [stim.target_z(qs[0]), stim.target_z(qs[1])], float(theta[k]))
        c.append_from_stim_program_text(("MPP(%g) " % q if q > 0 else "MPP ") + _checks_str())
        for i in range(4):
            tg = [stim.target_rec(-4 + i)] + ([stim.target_rec(-8 + i)] if r > 0 else [])
            c.append("DETECTOR", tg)
    c.append_from_stim_program_text("MPP " + _checks_str())              # perfect final readout
    for i in range(4):
        c.append("DETECTOR", [stim.target_rec(-4 + i), stim.target_rec(-8 + i)])
    c.append_from_stim_program_text("MPP X0*X3*X6")
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    return c

def sample(theta, q, R, shots, seed):
    det, obs = build_circuit(theta, q, R).compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return det, obs[:, 0].astype(np.uint8)

def build_matching(llr_float, q, R):
    """llr_float: 9 per-qubit weights ln((1-p)/p). Detector index = layer*4 + check."""
    m = pymatching.Matching()
    for r in range(R + 1):
        if r < R:
            for qb in range(N_Q):
                sig = qubit_sig(qb); cs = [c for c in range(4) if (sig >> c) & 1]; fid = {0} if qb in LOGICAL_SUPPORT else set()
                w = float(llr_float[qb])
                if len(cs) == 2: m.add_edge(r*4 + cs[0], r*4 + cs[1], fault_ids=fid, weight=w, merge_strategy="smallest-weight")
                else: m.add_boundary_edge(r*4 + cs[0], fault_ids=fid, weight=w, merge_strategy="smallest-weight")
        if r < R and q > 0:
            wt = math.log((1 - q) / q)
            for c in range(4): m.add_edge(r*4 + c, (r + 1)*4 + c, weight=wt, merge_strategy="smallest-weight")
        elif r < R:   # q = 0: forbid measurement edges with a large weight
            for c in range(4): m.add_edge(r*4 + c, (r + 1)*4 + c, weight=30.0, merge_strategy="smallest-weight")
    return m

def failures(m, det, obs):
    pred = m.decode_batch(det)[:, 0]
    return (pred != obs)

# ---------- measurement-aware estimator ----------
POP = np.array([bin(u).count("1") for u in range(16)])
def make_nll(qe):
    cu = (1 - 2*qe) ** POP
    def f(theta, phat, theta0, lam):
        A = 1.0 - 2.0*theta[None, :]*PAR; Ph = np.prod(A, axis=1)
        P = np.maximum((WHT @ (Ph*cu))/16.0, 1e-300)
        nll = -(phat*np.log(P)).sum() + 0.5*lam*((theta - theta0)**2).sum()
        dPh = (-2.0*PAR) * (Ph[:, None]/A)
        dP = (WHT @ (dPh*cu[:, None]))/16.0
        return nll, -(phat/P) @ dP + lam*(theta - theta0)
    return f

def fit_map_q(hist16, qe, lam=0.05, maxiter=200):
    th0 = theta_prior(); phat = hist16/hist16.sum()
    r = minimize(make_nll(qe), th0.copy(), args=(phat, th0, lam), jac=True, method="L-BFGS-B",
                 bounds=[(1e-5, 0.45)]*M, options={"maxiter": maxiter})
    return r.x

def mid_layer_hist(det, R, W):
    """16-bin histogram of detection-event vectors from layers 1..R-1 (0-based layers 1..R-1 have data errors and two
    measurement contributions). Uses the first ceil(W/(R-1)) shots, truncated to W events."""
    n = int(np.ceil(W/(R - 1))); d = det[:n].reshape(n, R + 1, 4)[:, 1:R, :]
    s = (d.astype(np.uint8) << np.arange(4)).sum(2).reshape(-1)[:W]
    return np.bincount(s.astype(int), minlength=16).astype(float)
