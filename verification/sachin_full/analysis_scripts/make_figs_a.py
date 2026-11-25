import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
df = pd.read_csv("results_a2.csv")
Ws = sorted(df.W.unique()); Ds = sorted(df.D.unique())
cols = {64: "#c0392b", 256: "#e08e0b", 1024: "#5b8c3a", 4096: "#1f6f8b", 10000: "#5b3f8c"}
fig, axs = plt.subplots(1, len(Ds), figsize=(10.2, 3.6), sharey=True)
for ax, D in zip(axs, Ds):
    for W in Ws:
        g = df[(df.W == W) & (df.D == D)].sort_values("T_b")
        ax.plot(g.T_b, g.rho_mean, color=cols[W], lw=1.8, marker="o", ms=2.5)
        ax.fill_between(g.T_b, g.ci_lo, g.ci_hi, color=cols[W], alpha=0.15, lw=0)
        ax.text(g.T_b.iloc[-1] * 1.08, g.rho_mean.iloc[-1], f"W={W}", color=cols[W], fontsize=7, va="center")
    for x, lab in [(60, "60 us"), (1084, "repo test\n(1,084)"), (10000, "10 ms")]:
        ax.axvline(x, color="k", lw=0.7, ls=":"); ax.text(x * 1.05, -0.17, lab, fontsize=7, va="bottom")
    ax.axhline(0.5, color="k", lw=0.5, ls="--"); ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log"); ax.set_xlim(40, 1.2e6); ax.set_ylim(-0.2, 1.05)
    ax.set_title(f"commit delay D = {D} rounds ({D/1000:g} ms)", fontsize=9, loc="left"); ax.set_xlabel("burst duration (rounds, 1 round = 1 us)")
axs[0].set_ylabel("share of the static-to-optimal gap recovered")
fig.tight_layout(); fig.savefig("fig_tracking.png"); plt.close(fig)

# break-even table: smallest T_b with rho >= 0.5 (log-linear interpolation)
def t_half(g, level=0.5):
    g = g.sort_values("T_b"); x = np.log(g.T_b.values); y = g.rho_mean.values
    for i in range(1, len(y)):
        if y[i] >= level and y[i-1] < level:
            return float(np.exp(x[i-1] + (level - y[i-1]) * (x[i] - x[i-1]) / (y[i] - y[i-1])))
    return float("nan") if y.max() < level else float(g.T_b.values[0])
rows = []
for W in Ws:
    for D in Ds:
        g = df[(df.W == W) & (df.D == D)]
        rows.append((W, D, t_half(g, 0.5), t_half(g, 0.9), g[g.T_b == 60].rho_mean.iloc[0], g[g.T_b == 1000].rho_mean.iloc[0], g[g.T_b == 10000].rho_mean.iloc[0]))
out = pd.DataFrame(rows, columns=["W", "D", "T_half(rho=0.5)", "T_90(rho=0.9)", "rho@60", "rho@1000", "rho@10000"])
out.to_csv("breakeven_a.csv", index=False); pd.set_option("display.width", 200); print(out.round(3).to_string(index=False))
