"""Task 2: robustness sweep of the tracking/latency analysis.
Scenarios vary burst strength, location, kind (single-only / pair-only) and shape (step vs ramped).
Two decision rules from the same MAP fit: AVID pair projection ('proj') and exact pair-aware MAP table ('joint')."""
import numpy as np, csv, time, sys
from avid_model import *; from avid_decoders import *
PRIOR = theta_prior()
def mk(**kw):
    t = PRIOR.copy()
    for k, v in kw.items():
        t[eval(k)] = v
    return t
SCEN = {
 "repo (q4 .055, pair45 .075)":  mk(**{"idx_single(4)": .055, "idx_edge(4,5)": .075}),
 "weak (q4 .03, pair45 .04)":    mk(**{"idx_single(4)": .03,  "idx_edge(4,5)": .04}),
 "strong (q4 .09, pair45 .12)":  mk(**{"idx_single(4)": .09,  "idx_edge(4,5)": .12}),
 "top (q1 .055, pair01 .075)":   mk(**{"idx_single(1)": .055, "idx_edge(0,1)": .075}),
 "bottom (q7 .055, pair78 .075)":mk(**{"idx_single(7)": .055, "idx_edge(7,8)": .075}),
 "single only (q3 .075)":        mk(**{"idx_single(3)": .075}),
 "pair only (pair45 .075)":      mk(**{"idx_edge(4,5)": .075}),
}
WS = [64, 256, 1024, 4096, 10000]; DS = [0, 1500, 6000]
TB = [60, 120, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
FG = np.linspace(0, 1, 11); ND = 150; NTR = 300
rng = np.random.default_rng(11)
Pb = dp_joint(PRIOR); STATIC = map_decision(Pb); O0 = ler_opt(Pb); S0 = ler_of(Pb, STATIC)

def draw_losses(thb, Pm_b, Pj_b):
    """LB[dec][W][fi, draw], L0[...]: exact per-round LER of the sampled decision table under burst / baseline noise."""
    Pm0 = p_syndrome(PRIOR); out = {d: {} for d in ("proj", "joint")}; cache = {}
    for W in WS:
        LB = {d: np.zeros((len(FG), ND)) for d in out}; L0 = {d: np.zeros((len(FG), ND)) for d in out}
        for fi, f in enumerate(FG):
            nb = int(round(f*W))
            for j in range(ND):
                h = rng.multinomial(nb, Pm_b) + rng.multinomial(W-nb, Pm0)
                th = fit_map(h.astype(float))
                for d, tab in (("proj", minweight_table(projected_llr_q78(th, PRIOR))), ("joint", joint_table(th))):
                    key = tuple(int(x) for x in tab)
                    if key not in cache: cache[key] = (ler_of(Pj_b, tab), ler_of(Pb, tab))
                    LB[d][fi, j], L0[d][fi, j] = cache[key]
        for d in out: out[d][W] = (LB[d], L0[d])
    return out

def tri_cum(x, T):
    """integral of triangle alpha over [0,x], alpha(u)=2u/T on [0,T/2], 2(T-u)/T on [T/2,T]; x clipped to [0,T]."""
    x = np.clip(x, 0, T); h = T/2
    a = np.minimum(x, h); first = a*a/T                         # int 2u/T = u^2/T
    b = np.maximum(x - h, 0); second = (T - h)*0 + (2*b - (2*(h*b + b*b/2) - 2*h*b - 0)/1.0*0)  # placeholder replaced below
    # integral over [h, h+b] of 2(T-u)/T du = (2/T)*(T*b - (( (h+b)^2 - h^2)/2)) = 2b - ((h+b)^2-h^2)/T
    second = 2*b - (((h+b)**2) - h*h)/T
    return first + second

def one_trial(LB, L0, SB, O_B, W, D, Tb, shape):
    t0 = 2*W + rng.uniform(0, W); t1 = t0 + Tb; H1 = t1 + 3*W + D
    ek = np.arange(1, int(np.ceil(H1/W)) + 1) * W
    A = (lambda t: np.clip(t - t0, 0, Tb)) if shape == "step" else (lambda t: tri_cum(t - t0, Tb))
    f = (A(ek) - A(ek - W)) / W
    fi = np.clip(np.rint(f*(len(FG)-1)).astype(int), 0, len(FG)-1)
    j = rng.integers(0, ND, len(ek)); lb, l0 = LB[fi, j], L0[fi, j]
    commit = ek + D
    starts = np.concatenate([[0.0], commit]); ends = np.concatenate([commit, [max(H1, commit[-1] + 1)]])
    ok = starts < H1; starts, ends = starts[ok], np.minimum(ends[ok], H1)
    LBi = np.concatenate([[SB], lb])[ok]; L0i = np.concatenate([[S0], l0])[ok]
    ab = A(ends) - A(starts)
    Ead = (ab*LBi + ((ends-starts) - ab)*L0i).sum()
    ab_tot = A(H1) - A(0.0)
    Est = ab_tot*SB + (H1 - ab_tot)*S0; Eor = ab_tot*O_B + (H1 - ab_tot)*O0
    return (Est - Ead) / (Est - Eor)

if __name__ == "__main__":
    rows = []; summ = []; t_start = time.time()
    for name, thb in SCEN.items():
        Pm_b, Pj_b = p_syndrome(thb), dp_joint(thb)
        SB, O_B = ler_of(Pj_b, STATIC), ler_opt(Pj_b)
        L = draw_losses(thb, Pm_b, Pj_b)
        summ.append((name, SB, O_B, ler_of(Pj_b, minweight_table(marginal_llr_q78(thb)))))
        print(f"[{time.time()-t_start:4.0f}s] {name}: static {100*SB:.2f}% oracle {100*O_B:.2f}%", flush=True)
        for dec in ("proj", "joint"):
            for shape in ("step", "ramp"):
                for W in WS:
                    LB, L0 = L[dec][W]
                    for D in DS:
                        for Tb in TB:
                            r = np.array([one_trial(LB, L0, SB, O_B, W, D, Tb, shape) for _ in range(NTR)])
                            bs = [rng.choice(r, len(r)).mean() for _ in range(100)]
                            rows.append((name, dec, shape, W, D, Tb, r.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
    with open("results_sweep.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scenario", "dec", "shape", "W", "D", "T_b", "rho", "ci_lo", "ci_hi"]); w.writerows(rows)
    with open("sweep_gaps.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scenario", "static_LER_burst", "oracle_LER_burst", "marginal_oracleparam_LER_burst"]); w.writerows(summ)
    print("saved")
