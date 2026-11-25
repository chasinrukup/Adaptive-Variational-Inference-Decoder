"""Step 4: labeled calibration experiments at circuit level (R=4 noisy rounds, readout error q).
Labeled data = n_lab separate one-round calibration experiments (true logical bit recorded); they are extra rounds on top of
the W=4096 unlabeled detection events. Decoder = space-time matching with weights from the repo projection applied to the fitted theta."""
import numpy as np, csv, time, math
from avid_circuit import *; from avid_fix import *
PRIOR = theta_prior()
def mk(**kw):
    t = PRIOR.copy()
    for k, v in kw.items(): t[eval(k)] = v
    return t
SCEN = {"baseline (prior)": PRIOR.copy(),
        "R1 repo (q4 .055, pair45 .075)": mk(**{"idx_single(4)": .055, "idx_edge(4,5)": .075}),
        "R3 alias-swap (q4 .055, q3 .075)": mk(**{"idx_single(4)": .055, "idx_single(3)": .075})}
QS = [0.01, 0.02]; R = 4; W = 4096; NSEED = 20; NEVAL = 100000; NL = [0, 64, 256, 1024]
wl = lambda p: np.log((1 - p)/p)
rows = []; t0 = time.time()
for sname, th in SCEN.items():
    for q in QS:
        det, obs = sample(th, q, R, NEVAL, 1)
        run = lambda w: failures(build_matching(w, q, R), det, obs).mean()
        base = {"B0 static": run(wl(effective_p(PRIOR))), "MWPM true rates": run(wl(effective_p(th)))}
        samp_u = build_circuit(th, q, R).compile_detector_sampler(seed=900 + int(1000*q))
        samp_l = build_circuit(th, q, 1).compile_detector_sampler(seed=1900 + int(1000*q))
        n = int(np.ceil(W/(R - 1))); qe = 2*q*(1 - q); out = {}
        for seed in range(NSEED):
            d, _ = samp_u.sample(n, separate_observables=True); Hu = mid_layer_hist(d, R, W)
            dl, ol = samp_l.sample(max(NL), separate_observables=True)
            s_l = (dl[:, :4].astype(np.uint8) << np.arange(4)).sum(1).astype(int); l_l = ol[:, 0].astype(int)
            out.setdefault("AVID projection, naive estimator", []).append(run(projected_llr_q78(fit_map(Hu), PRIOR)/256.0))
            for nl in NL:
                Hl = np.zeros((16, 2)); np.add.at(Hl, (s_l[:nl], l_l[:nl]), 1)
                th_hat = fit_map32_q(Hl, Hu, q, qe)
                out.setdefault(f"AVID projection, aware + {nl} labeled", []).append(run(projected_llr_q78(th_hat, PRIOR)/256.0))
        res = {k: np.array(v) for k, v in out.items()}; res.update({k: np.array([v]) for k, v in base.items()})
        for k, v in res.items():
            se = v.std(ddof=1)/math.sqrt(len(v)) if len(v) > 1 else math.sqrt(v[0]*(1 - v[0])/NEVAL)
            rows.append((sname, q, k, v.mean(), se, len(v)))
        print(f"[{time.time()-t0:4.0f}s] {sname} q={q}: " + " | ".join(f"{k.replace('AVID projection, ','')[:26]} {100*v.mean():.2f}" for k, v in res.items()), flush=True)
with open("results_fix_circuit.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["scenario", "q", "decoder", "LER", "se", "n"]); w.writerows(rows)
