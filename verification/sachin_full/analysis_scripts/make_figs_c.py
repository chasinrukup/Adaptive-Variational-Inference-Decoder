import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
d = pd.read_csv("results_c.csv"); sc = list(dict.fromkeys(d.scenario))
style = {"B0 static": ("#7f7f7f", "o", "-"), "MWPM true rates": ("#2c3e50", "s", "--"),
         "AVID projection, naive estimator": ("#c0392b", "^", "-"), "AVID projection, aware estimator": ("#1f6f8b", "v", "-")}
fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.5), sharey=False)
for ax, s in zip(axs, sc):
    g = d[d.scenario == s]
    for k, (c, m, ls) in style.items():
        h = g[g.decoder == k].sort_values("q")
        if k.endswith("aware estimator") and len(h) < 4:   # q=0: aware == naive
            n0 = g[(g.decoder == "AVID projection, naive estimator") & (g.q == 0)]
            h = pd.concat([n0, h]).sort_values("q")
        ax.errorbar(h.q*100, h.LER*100, yerr=1.96*h.se*100, color=c, marker=m, ms=4, lw=1.6, ls=ls, label=k, capsize=2)
    ax.set_title(s, fontsize=8.5, loc="left"); ax.set_xlabel("measurement error rate q (%)"); ax.set_xticks([0, .5, 1, 2])
axs[0].set_ylabel("logical error rate over 4 rounds (%)"); axs[0].legend(fontsize=6.5, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("fig_circuit.png"); plt.close(fig)
