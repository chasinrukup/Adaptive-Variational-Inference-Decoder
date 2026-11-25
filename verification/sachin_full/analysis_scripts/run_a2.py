"""Timeline simulation: recovered fraction of the static-to-oracle gap for a burst of T_b rounds,
epoch/window W, commit delay D (all in rounds; 1 round = 1 us). Uses the decision tables sampled in run_a1.py."""
import numpy as np, pickle, csv, time
from avid_model import *; from avid_decoders import *

PRIOR = theta_prior()
burst = PRIOR.copy(); burst[idx_single(4)] = 0.055; burst[idx_edge(4,5)] = 0.075
Pj_base, Pj_burst = dp_joint(PRIOR), dp_joint(burst)

A1 = pickle.load(open("a1.pkl", "rb"))
inv = {v: np.array(k) for k, v in A1["tables"].items()}
NT = len(inv)
LER_B = np.array([ler_of(Pj_burst, inv[i]) for i in range(NT)])      # per-round LER of table i during a burst
LER_0 = np.array([ler_of(Pj_base, inv[i]) for i in range(NT)])       # ... and in baseline noise
static = map_decision(Pj_base)
S_B, S_0 = ler_of(Pj_burst, static), ler_of(Pj_base, static)
O_B, O_0 = ler_opt(Pj_burst), ler_opt(Pj_base)
FG = A1["FG"]; PROBS = A1["probs"]                                   # PROBS[W][fi, table_id]

# static decoder is not one of the sampled tables in general; give it a dedicated id
STATIC_ID = NT
LER_B = np.append(LER_B, S_B); LER_0 = np.append(LER_0, S_0)

rng = np.random.default_rng(7)

def overlap(a0, a1, b0, b1):
    return np.maximum(0.0, np.minimum(a1, b1) - np.maximum(a0, b0))

def one_trial(W, D, Tb):
    t0 = 2*W + rng.uniform(0, W); t1 = t0 + Tb
    H1 = t1 + 2*W + D + W                                          # horizon: burst plus stale rounds after it
    n_ep = int(np.ceil(H1 / W)) + 1
    ek = np.arange(1, n_ep) * W                                    # epoch boundaries (window ends)
    f = overlap(ek - W, ek, t0, t1) / W
    fi = np.clip(np.rint(f * (len(FG) - 1)).astype(int), 0, len(FG) - 1)
    p = PROBS[W][fi]                                               # (n_ep-1, NT)
    cum = p.cumsum(axis=1); u = rng.random(len(ek))[:, None]
    tab = (u > cum).sum(axis=1).clip(0, NT - 1)                    # sampled table per epoch
    commit = ek + D                                                # table takes effect here
    starts = np.concatenate([[0.0], commit]); ends = np.concatenate([commit, [max(H1, commit[-1] + 1)]])
    ids = np.concatenate([[STATIC_ID], tab])
    keep = starts < H1
    starts, ends, ids = starts[keep], np.minimum(ends[keep], H1), ids[keep]
    lb = overlap(starts, ends, t0, t1); ln = (ends - starts) - lb
    E_ad = (lb * LER_B[ids] + ln * LER_0[ids]).sum()
    E_st = Tb * S_B + (H1 - Tb) * S_0
    E_or = Tb * O_B + (H1 - Tb) * O_0
    return (E_st - E_ad) / (E_st - E_or)

if __name__ == "__main__":
    WS = [64, 256, 1024, 4096, 10000]; DS = [0, 1500, 6000]
    TB = [60, 120, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
    NTR = 500; rows = []; t_start = time.time()
    for W in WS:
        for D in DS:
            for Tb in TB:
                r = np.array([one_trial(W, D, Tb) for _ in range(NTR)])
                bs = [rng.choice(r, len(r)).mean() for _ in range(300)]
                rows.append((W, D, Tb, r.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), r.std()))
        print(f"[{time.time()-t_start:4.0f}s] W={W} done", flush=True)
    with open("results_a2.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["W", "D", "T_b", "rho_mean", "ci_lo", "ci_hi", "rho_sd"]); w.writerows(rows)
    print("saved results_a2.csv")
