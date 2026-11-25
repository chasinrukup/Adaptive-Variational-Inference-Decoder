"""X+Z idea. (1) Identifiability under the repo's independent X/Z model. (2) Coupled-hotspot model: the X-side rates of the hot
elements are raised by r times their Z excess; a profile likelihood-ratio test on the X histogram tries to tell pair from single."""
import numpy as np, csv
from avid_model import *; from avid_decoders import *
SIGX = np.array([0x1,0x1,0x2,0x5,0x9,0xA,0x4,0x8,0x8, 0x0,0x3,0xC,0x3,0xC,0x0, 0x4,0x1,0x8,0x1,0x8,0x2])   # repo NM_EVENT_SIGNATURES_Z_BASIS
# cross-check against the Z-check layout z0={0,1,3,4}, z1={2,5}, z2={3,6}, z3={4,5,7,8}
ZCH = [(0,1,3,4),(2,5),(3,6),(4,5,7,8)]
qx = lambda q: sum(1<<c for c, sup in enumerate(ZCH) if q in sup)
chk = [qx(q) for q in range(9)] + [qx(i)^qx(j) for i, j in EDGES]
assert list(SIGX) == chk, (list(SIGX), chk)
PARX = np.array([[bin(u & int(SIGX[k])).count("1") & 1 for k in range(M)] for u in range(16)])
px = lambda th: (WHT @ np.prod(1.0 - 2.0*th[None, :]*PARX, axis=1))/16.0

# ---- (1) identifiability under independent X/Z ----
th0 = np.random.default_rng(0).uniform(0.005, 0.05, M); thx0 = np.random.default_rng(1).uniform(0.003, 0.03, M)
joint = lambda tz: np.outer(p_syndrome(tz), px(thx0)).reshape(-1)          # 256-bin P(sx, sz) with independent bases
def jac(f, x):
    return np.array([(f(x+1e-6*np.eye(M)[k]) - f(x-1e-6*np.eye(M)[k]))/2e-6 for k in range(M)]).T
rk = lambda J: int((np.linalg.svd(J, compute_uv=False) > 1e-6*np.linalg.svd(J, compute_uv=False)[0]).sum())
print("rank of dP/d(theta_Z): Z histogram alone = %d, joint 256-bin (independent X/Z) = %d"%(rk(jac(p_syndrome, th0)), rk(jac(joint, th0))))

# ---- (2) coupled hotspot ----
PRIOR = theta_prior(); BASEX = np.full(M, 0.002); BASEX[:9] = 0.005
a_lo = 0.012; PROD = (1-2*0.012)*(1-2*0.075); a_hi = (1-PROD/(1-2*0.004))/2
def zparams(kind):
    t = PRIOR.copy(); t[idx_single(4)] = 0.055
    if kind == "pair": t[idx_edge(4,5)] = 0.075
    else: t[idx_single(3)] = a_hi
    return t
assert np.abs(p_syndrome(zparams("pair")) - p_syndrome(zparams("single"))).max() < 1e-12      # identical Z histogram
def xparams(kind, r):
    ex = zparams(kind) - PRIOR; return np.clip(BASEX + r*ex, 1e-5, 0.45)
rng = np.random.default_rng(2026); RS = [0.0, 0.1, 0.25, 0.5, 1.0]; WS = [256, 1024, 4096]; NT = 2000; GRID = np.linspace(0, 1.5, 31); TAU = np.log(20)
PX = {k: np.array([px(xparams(k, r)) for r in GRID]) for k in ("pair", "single")}     # model histograms on the profile grid
LER = {"pair": dict(static=8.83, proj=4.10), "single": dict(static=3.27, proj=10.34)}     # exact values from the trap curve (x=0 and x=1)
rows = []
for r in RS:
    for W in WS:
        out = {}
        for truth in ("pair", "single"):
            p = px(xparams(truth, r)); H = rng.multinomial(W, p, size=NT).astype(float)
            ll = {k: (H @ np.log(np.maximum(PX[k], 1e-300)).T).max(axis=1) for k in PX}
            llr = ll["pair"] - ll["single"]
            out[truth] = ((llr > TAU).mean(), (llr < -TAU).mean())
        pp, ps = out["pair"][0], out["single"][0]                                   # P(decide pair | truth)
        e_pair = pp*LER["pair"]["proj"] + (1-pp)*LER["pair"]["static"]
        e_single = ps*LER["single"]["proj"] + (1-ps)*LER["single"]["static"]
        rows.append((r, W, pp, out["pair"][1], ps, out["single"][1], e_pair, e_single))
        print(f"r={r:4.2f} W={W:5d} | truth=pair: decide pair {pp:5.2f} single {out['pair'][1]:5.2f} | truth=single: decide pair {ps:5.2f} single {out['single'][1]:5.2f} | expected LER pair {e_pair:5.2f}%  single {e_single:5.2f}%")
with open("results_xz.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["r", "W", "P_pair_given_pair", "P_single_given_pair", "P_pair_given_single", "P_single_given_single", "expected_LER_pair_pct", "expected_LER_single_pct"]); w.writerows(rows)
