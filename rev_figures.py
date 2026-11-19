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


def _drawn_points(ax):
    """Display coordinates of everything plotted in an axes (line vertices and points along
    the segments, scatter points, error-bar segments)."""
    pts = []
    for ln in ax.get_lines():
        d = ln.get_transform().transform(ln.get_xydata())
        d = d[np.isfinite(d).all(axis=1)]
        if len(d) > 1:
            d = np.vstack([d, np.linspace(d[:-1], d[1:], 14).reshape(-1, 2)])
        pts.append(d)
    for col in ax.collections:
        if hasattr(col, "get_segments") and len(col.get_segments()):
            for seg in col.get_segments():
                d = col.get_transform().transform(seg)
                if len(d) > 1:
                    d = np.vstack([d, np.linspace(d[:-1], d[1:], 14).reshape(-1, 2)])
                pts.append(d)
        elif len(col.get_offsets()):
            pts.append(col.get_offset_transform().transform(col.get_offsets()))
    pts = [d for d in pts if len(d)]
    return np.vstack(pts) if pts else np.empty((0, 2))


def audit(fig, name, boxed=()):
    """Layout check before a figure is saved. It raises, so that a faulty figure cannot be
    written, if
      - a text of a (text, box) pair leaves its box,
      - two texts overlap,
      - a text or a legend placed inside an axes crosses the frame of that axes,
      - a text or a legend placed inside an axes covers plotted data."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    problems = []
    inside = lambda t, b, m=0.0: (b.x0 - m <= t.x0 and t.x1 <= b.x1 + m and
                                  b.y0 - m <= t.y0 and t.y1 <= b.y1 + m)
    for text, box in boxed:
        if not inside(matplotlib.text.Text.get_window_extent(text, r),
                      box.get_window_extent(r)):
            problems.append(f"text {text.get_text()[:30]!r} leaves its box")
    # the texts that are actually drawn: titles, axis labels, annotations, legend entries
    # and the labels of the ticks inside the view (matplotlib also keeps labels of ticks
    # outside the view, which are never drawn)
    texts = list(fig.texts)
    for leg in fig.legends:
        texts += leg.get_texts()
    for ax in fig.axes:
        texts += list(ax.texts) + [ax.title, ax._left_title, ax._right_title]
        if ax.get_legend() is not None:
            texts += ax.get_legend().get_texts()
        if not ax.axison:
            continue
        texts += [ax.xaxis.label, ax.yaxis.label]
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            span = abs(hi - lo) * 1e-9
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if lo - span <= tick.get_loc() <= hi + span:
                    texts.append(tick.label1)
        # annotations and legends inside the axes: within the frame, and not on the data
        frame, pts = ax.get_window_extent(r), _drawn_points(ax)
        items = [(t.get_text()[:30], matplotlib.text.Text.get_window_extent(t, r))
                 for t in ax.texts if t.get_visible() and t.get_text().strip()]
        if ax.get_legend() is not None:
            items.append(("legend", ax.get_legend().get_window_extent(r)))
        for label, e in items:
            if not inside(e, frame):
                problems.append(f"{label!r} crosses the frame of its axes")
            if len(pts):
                hit = ((pts[:, 0] > e.x0 + 1) & (pts[:, 0] < e.x1 - 1) &
                       (pts[:, 1] > e.y0 + 1) & (pts[:, 1] < e.y1 - 1))
                if hit.any():
                    problems.append(f"{label!r} covers plotted data")
    texts = [t for t in texts if t.get_visible() and t.get_text().strip()]
    ext = [matplotlib.text.Text.get_window_extent(t, r) for t in texts]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = ext[i], ext[j]
            if min(a.width, a.height, b.width, b.height) <= 0:
                continue
            if a.x0 + 1 < b.x1 and b.x0 + 1 < a.x1 and a.y0 + 1 < b.y1 and b.y0 + 1 < a.y1:
                problems.append(f"texts overlap: {texts[i].get_text()[:25]!r} and "
                                f"{texts[j].get_text()[:25]!r}")
    if problems:
        raise RuntimeError(f"{name}: " + "; ".join(problems))
    print(f"{name}: layout check passed ({len(texts)} texts, {len(boxed)} boxed)")


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
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.955), ncol=2,
               fontsize=7.5, frameon=False)
    fig.align_ylabels(axes[:, 0])
    audit(fig, "fig_calibration")
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
    bands = []
    for ax, (y, label, log), tag in zip(axes, panels, "abc"):
        bands.append(ax.axvspan(fail_from, lv.max() + 0.01, color="0.5", alpha=0.15, lw=0))
        ax.plot(lv, y, "-o", color=INK, lw=1.3, ms=4)
        if log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(label, fontsize=8)
        ax.text(0.015, 0.9, f"({tag})", transform=ax.transAxes, fontsize=9, va="top")
    band_label = axes[0].text((fail_from + lv.max() + 0.01) / 2, 12.0,
                              "mean\ncontroller\nfails", ha="center", va="center",
                              fontsize=7, linespacing=1.15)
    i = int(np.argmax(panels[1][0]))
    axes[1].annotate(f"{panels[1][0][i]:.0f} at {lv[i]:.2f} rad,\nmean controller still holds",
                     xy=(lv[i], panels[1][0][i]), xytext=(0.115, 60), fontsize=8,
                     ha="center", va="center",
                     arrowprops=dict(arrowstyle="-", color="0.4", lw=0.7))
    axes[2].set_xlabel("held set-point [rad]")
    axes[2].set_xlim(lv.min() - 0.01, lv.max() + 0.01)
    fig.align_ylabels(axes)
    audit(fig, "fig_boundary", [(band_label, bands[0])])
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
    audit(fig, "fig_lambda")
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
    ax.set_ylim(-0.66, -0.27)
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
    ax.set_xlabel("offset from the bottom")
    ax.set_yticks([0.3, 1, 3]); ax.set_yticks([], minor=True)
    ax.set_yticklabels(["0.3", "1", "3"])
    ax.set_ylabel("spread of $J$")
    ax.legend(fontsize=6.5, loc="center left", framealpha=0.9, bbox_to_anchor=(0.0, 0.62))
    ax.set_title("(c) towards the force limit", fontsize=8, loc="left")
    for ax in axes:
        ax.tick_params(labelsize=7)
    audit(fig, "fig_mountaincar")
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
    ax.set_title("1  Evolve an uncertainty\nfor every weight", fontsize=8.5, loc="left")
    labels = ["trained neurocontroller\n" r"weights $\mu^\star$, found by a GA",
              "genetic algorithm evolves one\n" r"standard deviation $\sigma_i$ per weight" "\n"
              r"fitness: negative ELBO, $\mathrm{E}[J]+\lambda\,\mathrm{KL}$",
              "posterior over controllers\n" r"$w_i\sim\mathcal{N}(\mu^\star_i,\sigma_i^2)$"]
    # The boxes are sized from the rendered text, never the other way round: measure the
    # three labels, shrink the font until the widest one fits the panel, then draw boxes
    # of that common width around them.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    padx, pady, gap, fs = 0.05, 0.04, 0.11, 7.6
    while True:
        txts = [ax.text(0.5, 0.5, t, ha="center", va="center", fontsize=fs, linespacing=1.25)
                for t in labels]
        ext = [t.get_window_extent(rend).transformed(ax.transData.inverted()) for t in txts]
        width = max(e.width for e in ext) + 2 * padx
        total = sum(e.height + 2 * pady for e in ext) + gap * (len(ext) - 1)
        if width <= 1.0 and total <= 1.0:
            break
        for t in txts:
            t.remove()
        fs -= 0.2
    boxed, top = [], 1.0 - (1.0 - total) / 2
    for t, e in zip(txts, ext):
        h = e.height + 2 * pady
        t.set_position((0.5, top - h / 2))
        box = FancyBboxPatch((0.5 - width / 2, top - h), width, h,
                             boxstyle="round,pad=0,rounding_size=0.02", fc="0.96", ec="0.3",
                             lw=0.8)
        ax.add_patch(box); boxed.append((t, box))
        if len(boxed) > 1:                              # arrow from the box above
            ax.add_patch(FancyArrowPatch((0.5, top + gap - 0.012), (0.5, top + 0.012),
                                         arrowstyle="-|>", mutation_scale=9, color="0.3",
                                         lw=0.9))
        top -= h + gap

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
    ax.annotate("controller fails", xy=(0.35, 0.03), xytext=(0.185, 0.03), fontsize=7,
                ha="center", va="center", arrowprops=dict(arrowstyle="-", color="0.4", lw=0.7))
    ax.annotate("warning one\nstep earlier", xy=(0.30, 861), xytext=(0.13, 120), fontsize=7,
                ha="center", va="center", arrowprops=dict(arrowstyle="-", color="0.4", lw=0.7))
    ax.set_title("3  The uncertainty warns\nbefore the controller fails", fontsize=8.5, loc="left")
    for ax in axes[1:]:
        ax.tick_params(labelsize=7)
    audit(fig, "graphical_abstract", boxed)
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
