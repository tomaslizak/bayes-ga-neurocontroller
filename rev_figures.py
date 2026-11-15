"""Figures of the revised manuscript.

fig_calibration.pdf   physics perturbation, 2048 posterior samples: robust spread of the
                      cost and share of samples with a fallen episode, evolved against
                      uniform sigma (replaces the K = 64 standard-deviation figure)
fig_boundary.pdf      capability-boundary sweep, 2048 samples, three stacked panels with one
                      quantity each (reviewer 3, comment 7)
fig_lambda.pdf        sensitivity to the tempering weight lambda (reviewer 1, comment 2)
fig_mountaincar.pdf   second benchmark: tracking, importance map, hold sweep

Run:   python rev_figures.py
Inputs: results_rev/calibration_K2048.json, results_rev/runs/*.json, results_rev/mct_*
Output: paper/figures/ and paper/graphical_abstract.*
"""

import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9, "font.family": "serif", "mathtext.fontset": "cm",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
OUT = "paper/figures"
RES = "results_rev"
INK, MAP, UNI = "#1a1a1a", "#1f77b4", "#ff7f0e"     # palette checked for colour-vision safety
KW_MAP = dict(color=MAP, marker="o", ms=3.5, lw=1.3, ls="-", label=r"evolved $\sigma$")
KW_UNI = dict(color=UNI, marker="s", ms=3.5, lw=1.3, ls="--", label=r"uniform $\sigma$")


def fig_calibration():
    d = json.load(open(f"{RES}/calibration_K2048.json"))["pendulum_seed0_lam0.003"]
    titles = {"g": "gravity $g$", "m": "mass $m$", "l": "length $l$",
              "max_torque": r"motor torque $u_{\max}$"}
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.6), sharex=True, sharey="row",
                             gridspec_kw=dict(height_ratios=[1.5, 1], hspace=0.12, wspace=0.08))
    for j, (p, title) in enumerate(titles.items()):
        rows = d["perturbation"][p]
        s = [r["scale"] for r in rows]
        for i, stat in enumerate(("spread", "any_fall")):
            ax = axes[i, j]
            ax.axvline(1.0, color="0.45", lw=0.7, ls=":")
            ax.plot(s, [r["map"][stat] for r in rows], **KW_MAP)
            ax.plot(s, [r["uni"][stat] for r in rows], **KW_UNI)
        axes[0, j].set_yscale("log")
        axes[0, j].set_title(title, fontsize=9)
        axes[1, j].set_xlabel("multiplier")
        axes[1, j].set_ylim(-0.06, 1.06)
    axes[0, 0].set_ylabel("spread of the cost $J$\nover posterior samples", fontsize=8)
    axes[1, 0].set_ylabel("share of samples\nthat fall", fontsize=8)
    axes[0, 0].legend(fontsize=7, loc="upper left", framealpha=0.9)
    fig.align_ylabels(axes[:, 0])
    fig.savefig(f"{OUT}/fig_calibration.pdf"); plt.close(fig)
    print("fig_calibration.pdf")


def fig_boundary():
    hold = json.load(open(f"{RES}/calibration_K2048.json"))["pendulum_seed0_lam0.003"]["hold"]
    lv = np.array([h["level"] for h in hold])
    panels = [
        (np.array([h["det_cost"] for h in hold]), "cost $J$ of the\nmean controller", True),
        (np.array([h["map"]["spread"] for h in hold]),
         "predictive uncertainty:\nspread of $J$ over\nposterior samples", True),
        (np.array([h["map"]["any_fall"] for h in hold]), "share of posterior\nsamples that fall",
         False),
    ]
    fail_from = 0.31                      # the mean controller holds 0.30 rad, fails at 0.32
    fig, axes = plt.subplots(3, 1, figsize=(4.6, 4.9), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1, 0.7], hspace=0.12))
    for ax, (y, label, log), tag in zip(axes, panels, "abc"):
        ax.axvspan(fail_from, lv.max() + 0.01, color="0.5", alpha=0.15, lw=0)
        ax.plot(lv, y, "-o", color=INK, lw=1.3, ms=4)
        if log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(label, fontsize=8)
        ax.text(0.015, 0.9, f"({tag})", transform=ax.transAxes, fontsize=9, va="top")
    axes[0].text(0.345, 2.0, "mean controller\nfails", ha="center", va="center", fontsize=8)
    i = int(np.argmax(panels[1][0]))
    axes[1].annotate(f"{panels[1][0][i]:.0f} at {lv[i]:.2f} rad,\nmean controller still holds",
                     xy=(lv[i], panels[1][0][i]), xytext=(0.115, 60), fontsize=8,
                     ha="center", va="center",
                     arrowprops=dict(arrowstyle="-", color="0.4", lw=0.7))
    axes[2].set_xlabel("held set-point [rad]")
    axes[2].set_xlim(lv.min() - 0.01, lv.max() + 0.01)
    fig.align_ylabels(axes)
    fig.savefig(f"{OUT}/fig_boundary.pdf"); plt.close(fig)
    print("fig_boundary.pdf")


def fig_lambda():
    lams = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    recs = {l: [json.load(open(p)) for p in sorted(glob.glob(f"{RES}/runs/lam{l:g}_s*.json"))]
            for l in lams}
    floor = recs[lams[0]][0]["floor"]
    get = lambda k: (np.array([np.mean([r[k] for r in recs[l]]) for l in lams]),
                     np.array([np.std([r[k] for r in recs[l]]) for l in lams]))
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.95), gridspec_kw=dict(wspace=0.62))
    panels = [("rho", r"rank correlation $\rho(\sigma, c)$", 1.0, False),
              ("fell", "fall rate [%]", 100.0, False),
              ("E_cost", r"expected cost $\mathrm{E}[J]$", 1.0, True),
              ("sigma_most", r"$\sigma$, most critical decile", 1.0, True)]
    for ax, (key, label, mult, logy), tag in zip(axes, panels, "abcd"):
        m, s = get(key)
        ax.errorbar(lams, m * mult, yerr=s * mult, fmt="-o", color=INK, lw=1.2, ms=3.5,
                    capsize=2, elinewidth=0.8)
        ax.axvline(3e-3, color="0.45", lw=0.7, ls=":")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(label, fontsize=7.5)
        ax.set_xticks([1e-3, 1e-2, 1e-1])
        ax.tick_params(labelsize=7)
        ax.text(0.05, 0.93, f"({tag})", transform=ax.transAxes, fontsize=8, va="top")
    axes[0].set_ylim(-0.65, 0.0)
    axes[0].texts[0].set_position((0.05, 0.12))
    axes[2].axhline(floor, color="0.45", lw=0.8, ls="--")
    axes[2].text(3.2e-1, floor * 1.05, "floor", fontsize=6.5, ha="right", va="bottom")
    axes[2].set_yticks([15, 20, 30, 50, 70]); axes[2].set_yticks([], minor=True)
    axes[2].set_yticklabels(["15", "20", "30", "50", "70"])
    m, _ = get("sigma_most")
    ref = m[2] * np.sqrt(np.array(lams) / 3e-3)
    axes[3].plot(lams, ref, color="0.45", lw=0.9, ls="--")
    axes[3].text(1.6e-2, 0.0052, r"$\propto\sqrt{\lambda}$", fontsize=8, color="0.3")
    axes[3].set_yticks([0.003, 0.01, 0.03]); axes[3].set_yticks([], minor=True)
    axes[3].set_yticklabels(["0.003", "0.01", "0.03"])
    fig.savefig(f"{OUT}/fig_lambda.pdf"); plt.close(fig)
    print("fig_lambda.pdf")


def fig_mountaincar():
    import torch
    from ga_torch import mountaincar_track as T
    import rev_mc_track as R
    from rev_pendulum import DEV, spearman

    b = R.bench()
    sig = torch.load(f"{RES}/sigma/mct_lam0.003_s0.pt").to(DEV)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25), gridspec_kw=dict(wspace=0.42))

    # (a) predictive tracking on the held-out reference
    ref = T.get_reference(device=DEV)
    gen = torch.Generator(device=DEV).manual_seed(7)
    W = b.mu.unsqueeze(0) + sig.unsqueeze(0) * torch.randn(128, b.mu.shape[0], device=DEV,
                                                           generator=gen)
    tr = T.simulate(W, ref, R.INPUTS, b.scale, R.HIDDEN, episodes=1, init_seed=99,
                    return_traj=True)
    x = tr["x"].cpu().numpy(); k = np.arange(x.shape[1])
    ax = axes[0]
    ax.plot(k, x[:40].T, color=MAP, lw=0.4, alpha=0.25)
    ax.plot(k, ref.cpu().numpy(), color=INK, lw=1.0, ls="--", label="set-point")
    ax.plot(k, x.mean(0), color=MAP, lw=1.1, label="posterior mean")
    ax.set_xlabel("step"); ax.set_ylabel("position $x$")
    ax.set_ylim(-0.66, -0.345)
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
    ax.set_title("(a) predictive tracking", fontsize=8, loc="left")

    # (b) importance map
    s = sig.cpu().numpy(); c = b.crit
    ax = axes[1]
    ax.scatter(c + 1e-5, s, s=5, color=MAP, alpha=0.6, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("weight criticality $c_i$"); ax.set_ylabel(r"evolved std $\sigma_i$")
    ax.set_title(fr"(b) importance map ($\rho={spearman(s, c):.2f}$)", fontsize=8, loc="left")

    # (c) hold sweep
    hold = json.load(open(f"{RES}/calibration_K2048.json"))["mountaincar_seed0_lam0.003"]["hold"]
    lv = [h["level"] for h in hold]
    ax = axes[2]
    ax.axvline(T.HOLD_LIMIT, color="0.45", lw=0.7, ls=":")
    ax.plot(lv, [h["map"]["spread"] for h in hold], **KW_MAP)
    ax.plot(lv, [h["uni"]["spread"] for h in hold], **KW_UNI)
    ax.set_yscale("log")
    ax.text(T.HOLD_LIMIT - 0.006, 0.36, "holdable\nlimit", fontsize=6.5, ha="right", va="center")
    ax.set_xlabel("held offset from the bottom")
    ax.set_ylabel("spread of $J$")
    ax.legend(fontsize=6.5, loc="center left", framealpha=0.9, bbox_to_anchor=(0.0, 0.62))
    ax.set_title("(c) towards the force limit", fontsize=8, loc="left")
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.savefig(f"{OUT}/fig_mountaincar.pdf"); plt.close(fig)
    print(f"fig_mountaincar.pdf (128 samples, share leaving the valley "
          f"{tr['fell_rate'].mean().item():.3f}, largest spread of x over time "
          f"{x.std(0).max():.4f})")


def graphical_abstract():
    """Graphical abstract (separate submission item, not a figure of the manuscript)."""
    import torch
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from rev_pendulum import Bench, DEV, spearman

    b = Bench()
    sig = torch.load("results_pd/sigma_lam0.003_s0_best.pt").to(DEV).cpu().numpy()
    hold = json.load(open(f"{RES}/calibration_K2048.json"))["pendulum_seed0_lam0.003"]["hold"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7),
                             gridspec_kw=dict(wspace=0.5, width_ratios=[1.25, 1, 1]))

    ax = axes[0]
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    boxes = [(0.86, "trained neurocontroller\n" r"weights $\mu^\star$, found by a GA"),
             (0.50, "genetic algorithm evolves one\n" r"standard deviation $\sigma_i$ per weight" "\n"
                    r"fitness: negative ELBO, $\mathrm{E}[J]+\lambda\,\mathrm{KL}$"),
             (0.13, "posterior over controllers\n" r"$w_i\sim\mathcal{N}(\mu^\star_i,\sigma_i^2)$")]
    heights = [0.2, 0.3, 0.2]
    for (y, text), h in zip(boxes, heights):
        ax.add_patch(FancyBboxPatch((0.02, y - h / 2), 0.96, h, boxstyle="round,pad=0.01",
                                    fc="0.96", ec="0.3", lw=0.8))
        ax.text(0.5, y, text, ha="center", va="center", fontsize=7.2)
    for y0, y1 in ((0.755, 0.66), (0.345, 0.24)):
        ax.add_patch(FancyArrowPatch((0.5, y0), (0.5, y1), arrowstyle="-|>", mutation_scale=9,
                                     color="0.3", lw=0.9))
    ax.set_title("1  Evolve an uncertainty\nfor every weight", fontsize=8.5, loc="left")

    ax = axes[1]
    ax.scatter(b.crit + 1e-5, sig, s=5, color=MAP, alpha=0.6, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("how much the weight matters"); ax.set_ylabel(r"evolved $\sigma_i$")
    ax.set_title("2  Critical weights become\nprecise, the rest uncertain", fontsize=8.5, loc="left")

    ax = axes[2]
    lv = np.array([h["level"] for h in hold])
    ax.axvspan(0.31, lv.max() + 0.01, color="0.5", alpha=0.15, lw=0)
    ax.plot(lv, [h["map"]["spread"] for h in hold], "-o", color=INK, lw=1.3, ms=3.5)
    ax.set_yscale("log")
    ax.set_xlabel("commanded angle [rad]"); ax.set_ylabel("predictive uncertainty")
    ax.text(0.348, 0.03, "controller\nfails", ha="center", va="center", fontsize=7)
    ax.annotate("warning one\nstep earlier", xy=(0.30, 861), xytext=(0.13, 120), fontsize=7,
                ha="center", va="center", arrowprops=dict(arrowstyle="-", color="0.4", lw=0.7))
    ax.set_title("3  The uncertainty warns\nbefore the controller fails", fontsize=8.5, loc="left")
    for ax in axes[1:]:
        ax.tick_params(labelsize=7)
    fig.savefig("paper/graphical_abstract.pdf")
    fig.savefig("paper/graphical_abstract.png", dpi=300)
    plt.close(fig)
    print(f"graphical_abstract.pdf/.png (rho of the plotted map {spearman(sig, b.crit):.2f})")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_calibration()
    fig_boundary()
    fig_lambda()
    fig_mountaincar()
    graphical_abstract()
