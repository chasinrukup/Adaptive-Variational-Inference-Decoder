"""Task 4 driver: circuit-level (R=4 noisy rounds + perfect readout) comparison, 3 noise scenarios x 4 measurement-error rates."""
import numpy as np, csv, time, math
from avid_circuit import *
PRIOR = theta_prior()
def mk(**kw):
    t = PRIOR.copy()
    for k, v in kw.items(): t[eval(k)] = v
    return t
SCEN = {"baseline (prior)": PRIOR.copy(),
        "R1 repo (q4 .055, pair45 .075)": mk(**{"idx_single(4)": .055, "idx_edge(4,5)": .075}),
        "R3 alias-swap (q4 .055, q3 .075)": mk(**{"idx_single(4)": .055, "idx_single(3)": .075})}
QS = [0.0, 0.005, 0.01, 0.02]; R = 4; W = 4096; NSEED = 20; NEVAL = 100000
wl = lambda p: np.log((1 - p)/p)
rows = []; t0 = time.time()
for sname, th in SCEN.items():
    for q in QS:
        det, obs = sample(th, q, R, NEVAL, 1)
        run = lambda w: failures(build_matching(w, q, R), det, obs)
        F = {"B0 static": run(wl(effective_p(PRIOR))), "MWPM true rates": run(wl(effective_p(th)))}
        sampler = build_circuit(th, q, R).compile_detector_sampler(seed=500 + int(1000*q))
        n = int(np.ceil(W/(R - 1))); qe = 2*q*(1 - q); per_seed = {}
        for seed in range(NSEED):
            d, _ = sampler.sample(n, separate_observables=True); h = mid_layer_hist(d, R, W)
            fits = {"naive": fit_map(h), "aware": fit_map_q(h, qe)}
            for tag, thh in fits.items():
                if tag == "aware" and q == 0: continue
                per_seed.setdefault(f"AVID projection, {tag} estimator", []).append(run(projected_llr_q78(thh, PRIOR)/256.0))
                per_seed.setdefault(f"MAP marginal, {tag} estimator", []).append(run(marginal_llr_q78(thh)/256.0))
        res = {k: np.array([v.mean() for v in vs]) for k, vs in per_seed.items()}
        res.update({k: np.array([v.mean()]) for k, v in F.items()})
        for k, v in res.items():
            se = v.std(ddof=1)/math.sqrt(len(v)) if len(v) > 1 else math.sqrt(v[0]*(1 - v[0])/NEVAL)
            rows.append((sname, q, k, v.mean(), se, len(v)))
        st = res["B0 static"][0]
        print(f"[{time.time()-t0:4.0f}s] {sname} q={q}: " + " | ".join(f"{k.replace('AVID projection','AVID proj')[:22]} {100*v.mean():.2f}" for k, v in res.items()), flush=True)
with open("results_c.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["scenario", "q", "decoder", "LER", "se", "n"]); w.writerows(rows)
