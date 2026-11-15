"""Regenerate the paper figures from the Pendulum results (compute-light).
Outputs -> paper/figures/. Run after train_pendulum, train_sigma_pendulum (seed 0)
and perturbation_pendulum have produced results_pd/*.
"""

import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ga_torch import pendulum as P
from ga_torch.simulation import setup_precision

plt.rcParams.update({
    "font.size": 9, "font.family": "serif", "mathtext.fontset": "cm",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
setup_precision("high")
OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)

INPUTS, H, LAM = ("e", "theta", "theta_dot"), 8, "0.003"
ref = P.get_reference(device=DEV)
scale = P.scale_tensor(INPUTS, device=DEV)
mu = torch.load("results_pd/baseline_no_ie_best.pt").to(DEV)
sig = torch.load(f"results_pd/sigma_lam{LAM}_s0_best.pt").to(DEV)
battery = P.validation_battery(device=DEV)
t = np.arange(ref.shape[0]) * P.DT
r = ref.cpu().numpy()


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = a.argsort().argsort() * 1.0; rb = b.argsort().argsort() * 1.0
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra**2).sum() * (rb**2).sum()))


# Fig 1: deterministic tracking (held-out reference)
out = P.simulate(mu.unsqueeze(0), ref, INPUTS, scale, H, episodes=4,
                 init_seed=99, return_traj=True)
th = out["theta"].reshape(4, -1).mean(dim=0).cpu().numpy()
fig, ax = plt.subplots(figsize=(3.4, 2.1))
ax.plot(t, r, "k--", lw=1.0, label="set-point $r$ [rad]")
ax.plot(t, th, "C0", lw=1.0, label=r"pendulum angle $\theta$")
ax.set_xlabel("time [s]"); ax.set_ylabel("angle [rad]")
ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
fig.savefig(f"{OUT}/fig_tracking.pdf"); plt.close(fig)
print("fig_tracking.pdf")

# Fig 2: Bayesian predictive tracking band + torque + spread over time
torch.manual_seed(0)
Kb = 128
W = mu.unsqueeze(0) + sig.unsqueeze(0) * torch.randn(Kb, mu.shape[0], device=DEV)
outb = P.simulate(W, ref, INPUTS, scale, H, episodes=1, init_seed=99, return_traj=True)
yb = outb["theta"].cpu().numpy()
ub = outb["u"].cpu().numpy()
fig, (ax1, axu, ax2) = plt.subplots(3, 1, figsize=(3.4, 4.4), sharex=True,
                                    gridspec_kw={"height_ratios": [2.0, 1.3, 1]})
for k in range(0, Kb, 3):
    ax1.plot(t, yb[k], color="C0", lw=0.3, alpha=0.12)
ax1.plot(t, r, "k--", lw=1.0, label="set-point")
ax1.plot(t, yb.mean(0), "C0", lw=1.1, label="posterior mean")
ax1.set_ylabel("angle [rad]")
ax1.legend(loc="upper right", fontsize=7, framealpha=0.9)
for k in range(0, Kb, 3):
    axu.plot(t, ub[k], color="C2", lw=0.3, alpha=0.12)
axu.plot(t, ub.mean(0), "C2", lw=1.0, label="torque $u$")
axu.axhline(2.0, color="gray", ls="--", lw=0.8)
axu.axhline(-2.0, color="gray", ls="--", lw=0.8)
axu.set_ylim(-2.35, 2.35)
axu.set_ylabel(r"torque [N$\cdot$m]")
axu.legend(loc="upper right", fontsize=7, framealpha=0.9)
ystd = yb.std(0)
ax2.fill_between(t, 0, ystd, color="C3", alpha=0.3)
ax2.plot(t, ystd, "C3", lw=1.0)
ax2.set_ylabel("angle spread"); ax2.set_xlabel("time [s]")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_bayes_tracking.pdf"); plt.close(fig)
print(f"fig_bayes_tracking.pdf (spread transient/steady ratio "
      f"{ystd[(t>2)&(t<4)].mean()/max(ystd[(t>5.5)&(t<7)].mean(),1e-9):.1f}x)")
lvl = np.abs(r) > 1e-6
print(f"  torque: peak |u|={np.abs(ub).max():.2f}, "
      f"mean |u| while holding nonzero levels={np.abs(ub[:, lvl]).mean():.2f}, "
      f"mean |u| at zero level={np.abs(ub[:, ~lvl]).mean():.2f}")

# Fig 3: importance map (single column: scatter + histogram)
crit = P.weight_criticality(mu, battery, INPUTS, scale, H,
                            episodes=battery.shape[0], init_seed=555).cpu().numpy()
s = sig.cpu().numpy()
rho = spearman(s, crit)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.4, 3.6), gridspec_kw={"height_ratios": [2, 1]})
a1.scatter(crit + 1e-5, s, s=9, alpha=0.5, edgecolors="none", color="C0")
a1.set_xscale("log"); a1.set_xlabel("weight criticality $c_i$")
a1.set_ylabel(r"evolved std $\sigma_i$")
a1.set_title(fr"(a) importance map (Spearman $={rho:.2f}$)", fontsize=9)
a2.hist(s, bins=30, color="C0", alpha=0.8)
a2.set_xlabel(r"evolved $\sigma_i$"); a2.set_ylabel("count")
a2.set_title(r"(b) distribution of $\sigma_i$", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_importance_map.pdf"); plt.close(fig)
print(f"fig_importance_map.pdf (Spearman {rho:.3f})")

# Fig 4: calibration under native physics perturbation
pert = json.load(open("results_pd/perturbation.json"))["results"]
params = ["g", "m", "l", "max_torque"]
labels = {"g": "gravity $g$", "m": "mass $m$", "l": "length $l$",
          "max_torque": r"motor torque $u_{\max}$"}
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.1), sharey=True)
for ax, p in zip(axes, params):
    d = pert[p]
    ax.plot(d["scale"], d["unc_map"], "o-", ms=3, color="C0", label="importance-map $\\sigma$")
    ax.plot(d["scale"], d["unc_uni"], "s--", ms=3, color="C1", label="uniform $\\sigma$")
    ax.axvline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_title(labels[p], fontsize=8.5)
    ax.set_xlabel("multiplier")
axes[0].set_ylabel("predictive uncertainty")
axes[0].legend(fontsize=6.5, loc="lower left", framealpha=0.9)
fig.tight_layout(); # first-submission version (64 samples, standard deviation); the revised figure is
# drawn by rev_figures.py from results_rev/calibration_K2048.json
fig.savefig(f"{OUT}/fig_calibration_K64.pdf"); plt.close(fig)
print("fig_calibration_K64.pdf")

# Fig 5: capability boundary (hold-level sweep)
hs = json.load(open("results_pd/hold_sweep.json"))
lv, unc, det = np.array(hs["level"]), np.array(hs["unc"]), np.array(hs["det_cost"])
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.plot(lv, det + 0.1, "k^-", ms=3.5, label="mean-controller cost")
ax.plot(lv, unc + 0.1, "C3o-", ms=3.5, label="predictive uncertainty")
ax.set_yscale("log")
ax.axvspan(0.31, lv.max(), color="gray", alpha=0.12)
ax.set_xlabel("held set-point [rad]")
ax.set_ylabel("cost / uncertainty")
ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
fig.tight_layout(); # first-submission version; the revised figure is drawn by rev_figures.py
fig.savefig(f"{OUT}/fig_boundary_K64.pdf"); plt.close(fig)
print("fig_boundary_K64.pdf")
print("done")
