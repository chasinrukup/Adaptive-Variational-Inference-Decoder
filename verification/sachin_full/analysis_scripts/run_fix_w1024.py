import numpy as np, csv, time
from avid_model import *; from avid_decoders import *; from avid_stim import *; from avid_fix import *
PRIOR = theta_prior()
base = PRIOR.copy(); base[idx_single(4)] = 0.055
PROD = (1-2*0.012)*(1-2*0.075); a_lo, a_hi = 0.012, (1-PROD/(1-2*0.004))/2
W = 1024; NSEED = 100; NL = [0, 16, 32, 64, 128, 256, 512]; XS = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
STATIC = minweight_table(marginal_llr_q78(PRIOR))
rows = []; t0 = time.time()
for x in XS:
    a = a_lo + x*(a_hi - a_lo); b = (1 - PROD/(1 - 2*a))/2
    th = base.copy(); th[idx_single(3)] = a; th[idx_edge(4, 5)] = b
    Pt = dp_joint(th); orc = ler_opt(Pt); st = ler_of(Pt, STATIC)
    sampler = build_one_round(th).compile_detector_sampler(seed=4242 + int(100*x))
    res = {nl: [] for nl in NL}; proj = []; wrong = {nl: [] for nl in NL}
    for seed in range(NSEED):
        det, obs = sampler.sample(W, separate_observables=True)
        s = (det.astype(np.uint8) << np.arange(4)).sum(1).astype(int); l = obs[:, 0].astype(int)
        proj.append(ler_of(Pt, minweight_table(projected_llr_q78(fit_map(np.bincount(s, minlength=16).astype(float)), PRIOR))))
        for nl in NL:
            Hl, Hu = split_counts(s, l, nl)
            tab = map_decision(dp_joint(fit_map32(Hl, Hu)))
            res[nl].append(ler_of(Pt, tab))
    for nl in NL:
        r = np.array(res[nl]); rows.append((x, a, b, nl, "labeled-MAP", r.mean(), r.std(ddof=1)/np.sqrt(NSEED), st, orc))
    r = np.array(proj); rows.append((x, a, b, 0, "AVID projection", r.mean(), r.std(ddof=1)/np.sqrt(NSEED), st, orc))
    print(f"[{time.time()-t0:4.0f}s] x={x:.1f} static {100*st:.2f} oracle {100*orc:.2f} projection {100*np.mean(proj):.2f} | labeled n=0 {100*np.mean(res[0]):.2f} 64 {100*np.mean(res[64]):.2f} 256 {100*np.mean(res[256]):.2f}", flush=True)
with open("results_fix_w1024.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["x", "theta_Z3", "theta_pair45", "n_labeled", "method", "LER", "se", "static", "oracle"]); w.writerows(rows)
