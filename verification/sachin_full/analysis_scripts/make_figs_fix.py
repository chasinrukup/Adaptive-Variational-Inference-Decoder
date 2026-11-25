import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
d = pd.read_csv("results_fix.csv"); d2 = pd.read_csv("results_fix_w1024.csv")
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
ax = axs[0]; xs = sorted(d.x.unique()); g0 = d[d.method == "labeled-MAP"]
ax.plot(xs, [100*d[(d.x == x)].static.iloc[0] for x in xs], color="#7f7f7f", marker="o", ms=3, label="static (no adaptation)")
ax.plot(xs, [100*d[(d.x == x) & (d.method == "AVID projection")].LER.iloc[0] for x in xs], color="#c0392b", marker="^", ms=4, label="AVID projection")
for nl, c in ((64, "#e08e0b"), (256, "#1f6f8b"), (1024, "#5b8c3a")):
    ax.plot(xs, [100*g0[(g0.x == x) & (g0.n_labeled == nl)].LER.iloc[0] for x in xs], color=c, marker="s", ms=3, label=f"labeled MAP, {nl} of 4096 rounds labeled")
ax.plot(xs, [100*d[(d.x == x)].oracle.iloc[0] for x in xs], color="k", ls="--", lw=1, label="optimal")
ax.set_xlabel("share of the excess at syndrome 0x4 that is the pair (0) or a single qubit 3 (1)"); ax.set_ylabel("logical error rate (%)")
ax.set_title("Same histogram for every x, W = 4096", fontsize=9, loc="left"); ax.legend(frameon=False, fontsize=6.8, loc="upper center")
ax = axs[1]
for dd, W, c in ((d, 4096, "#1f6f8b"), (d2, 1024, "#5b3f8c")):
    l = dd[dd.method == "labeled-MAP"].copy(); l["reg"] = 100*(l.LER - l.oracle); t = l.groupby("n_labeled").reg.max()
    ax.plot(t.index.where(t.index > 0, 8), t.values, marker="o", ms=3, color=c, label=f"labeled MAP, W = {W}")
ax.axhline(100*(d.static - d.oracle).max(), color="#7f7f7f", ls="-", lw=1.2); ax.text(9, 4.95, "static", color="#7f7f7f", fontsize=8)
pr = d[d.method == "AVID projection"]; ax.axhline(100*(pr.LER - pr.oracle).max(), color="#c0392b", lw=1.2); ax.text(9, 7.2, "AVID projection", color="#c0392b", fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("labeled rounds (0 plotted at 8)"); ax.set_ylabel("worst-case excess error over optimal (points)")
ax.set_title("Worst case over all x", fontsize=9, loc="left"); ax.legend(frameon=False, fontsize=8, loc="center right")
fig.tight_layout(); fig.savefig("fig_fix.png"); plt.close(fig)
