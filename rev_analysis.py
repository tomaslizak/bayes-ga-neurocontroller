"""Revision analyses that need no training (work packages A8 and A9).

A9  Re-score the 56 sigma vectors behind the submitted paper with one common scorer at
    K = 2048 posterior samples, so E[J] and the fall rate are not dominated by MC noise.
A8  Sensitivity of the criticality diagnostic to the nudge: absolute delta over a range,
    and the relative variant delta * |w_i|. Reports the rank correlation of each variant
    with the evolved sigma (50 seeds on the main baseline) and with the delta = 0.05
    criticality used in the paper.
Also: curvature check. Under a quadratic approximation of the cost the ELBO optimum is
    sigma_i = sqrt(lambda / H_ii); H_ii is estimated by central differences.

Run:   python rev_analysis.py
Output: results_rev/analysis.json
"""

import json
import os

import numpy as np
import torch

from ga_torch import pendulum as P
from rev_pendulum import Bench, DEV, HIDDEN, INPUTS, spearman

LAM, SMIN, SMAX = 0.003, 1e-3, 0.3


def main_sigma_path(seed):
    legacy = f"results_pd/sigma_lam0.003_s{seed}_best.pt"
    return legacy if seed < 3 else f"results_pd/sigma_b0_s{seed}_best.pt"


@torch.no_grad()
def nudge_costs(bench, d):
    """Battery cost at mu, mu + d_i e_i and mu - d_i e_i for a per-weight nudge vector d."""
    mu = bench.mu
    n_w = mu.shape[0]
    eye = torch.diag(d.to(mu))
    batch = torch.cat([mu.unsqueeze(0), mu.unsqueeze(0) + eye, mu.unsqueeze(0) - eye], dim=0)
    cost = P.simulate(batch, bench.battery, INPUTS, bench.scale, HIDDEN, bench.E,
                      init_seed=555)["cost"]
    return cost[0], cost[1:n_w + 1], cost[n_w + 1:]


def criticality(bench, d):
    base, cp, cm = nudge_costs(bench, d)
    return (0.5 * ((cp - base).abs() + (cm - base).abs())).cpu().numpy()


def stats(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), std=float(v.std()), min=float(v.min()),
                max=float(v.max()))


def main():
    os.makedirs("results_rev", exist_ok=True)
    out = {}

    # ---------------- A9: common high-K re-scoring ----------------
    bench = Bench()
    main_recs, sig_main = [], []
    for s in range(50):
        sigma = torch.load(main_sigma_path(s)).to(DEV)
        r = bench.score(sigma); r["seed"] = s
        main_recs.append(r); sig_main.append(sigma.cpu().numpy())
    sig_main = np.stack(sig_main)                                    # (50, n_w)
    multi = []
    for b in (1, 2):
        bb = Bench(f"results_pd/baseline_b{b}.pt")
        for s in range(3):
            sigma = torch.load(f"results_pd/sigma_b{b}_s{s}_best.pt").to(DEV)
            r = bb.score(sigma); r.update(baseline=b, seed=s, floor=bb.floor)
            multi.append(r)
    std_sigma = torch.load("results_pd/sigma_lam0.003_s0_stdops_best.pt").to(DEV)
    out["rescore"] = dict(floor=bench.floor, K=2048, main=main_recs, multi=multi,
                          stdops_s0=bench.score(std_sigma))
    allr = main_recs + multi
    print(f"A9 re-score at K=2048 (floor {bench.floor:.3f})")
    for name, rs in (("main 50 seeds", main_recs), ("all 56 runs", allr)):
        rho = stats([r["rho"] for r in rs]); fell = stats([r["fell"] for r in rs])
        ec = stats([r["E_cost"] for r in rs])
        print(f"  {name:<14} rho {rho['mean']:+.3f}±{rho['std']:.3f} "
              f"[{rho['min']:+.3f},{rho['max']:+.3f}] | fell mean {fell['mean']:.5f} "
              f"max {fell['max']:.5f} | E[J] mean {ec['mean']:.2f} max {ec['max']:.2f}")
    for b in (1, 2):
        rs = [r for r in multi if r["baseline"] == b]
        rho = stats([r["rho"] for r in rs])
        print(f"  baseline b{b}    rho {rho['mean']:+.3f}±{rho['std']:.3f} floor {rs[0]['floor']:.2f}"
              f" | E[J] max {max(r['E_cost'] for r in rs):.2f} fell max {max(r['fell'] for r in rs):.5f}")
    so = out["rescore"]["stdops_s0"]
    print(f"  std-ops seed 0  rho {so['rho']:+.3f} fell {so['fell']:.4f} E[J] {so['E_cost']:.1f} "
          f"mean sigma {so['sigma_mean']:.3f}")
    s0 = main_recs[0]
    print(f"  self-adaptive seed 0  rho {s0['rho']:+.3f} fell {s0['fell']:.4f} "
          f"E[J] {s0['E_cost']:.2f} mean sigma {s0['sigma_mean']:.3f}")

    # ---------------- A8: delta sensitivity ----------------
    mu = bench.mu
    absw = mu.abs()
    print(f"\nA8 criticality nudge. |w|: median {absw.median():.3f}, "
          f"p10 {absw.quantile(0.1):.3f}, min {absw.min():.4f}")
    ref = bench.crit                                                  # delta = 0.05 absolute
    rows = []
    variants = [("abs", d) for d in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)] + \
               [("rel", d) for d in (0.02, 0.05, 0.1, 0.2)]
    for kind, d in variants:
        dvec = torch.full_like(mu, d) if kind == "abs" else d * absw
        c = criticality(bench, dvec)
        rho_sig = [spearman(sig_main[i], c) for i in range(50)]
        row = dict(kind=kind, delta=d, rho_sigma=stats(rho_sig),
                   rho_with_paper_crit=spearman(c, ref),
                   crit_p50=float(np.median(c)), crit_max=float(c.max()),
                   decades=float(np.log10(np.quantile(c, 0.95) / max(np.quantile(c, 0.05), 1e-12))))
        if kind == "rel":       # relative nudge against relative width sigma/|w|
            rel_sig = sig_main / np.maximum(absw.cpu().numpy(), 1e-6)
            row["rho_relsigma"] = stats([spearman(rel_sig[i], c) for i in range(50)])
        rows.append(row)
        extra = (f" | rho(sigma/|w|, c) {row['rho_relsigma']['mean']:+.3f}"
                 if kind == "rel" else "")
        print(f"  {kind} {d:<6} rho(sigma,c) {row['rho_sigma']['mean']:+.3f}±"
              f"{row['rho_sigma']['std']:.3f} | rank agreement with paper c "
              f"{row['rho_with_paper_crit']:+.3f} | p50 {row['crit_p50']:.3g} "
              f"max {row['crit_max']:.3g}{extra}")
    out["delta"] = rows

    # ---------------- curvature check ----------------
    print("\nCurvature check: sigma_pred = clip(sqrt(lambda / H_ii))")
    curv = []
    for d in (0.01, 0.02, 0.05):
        base, cp, cm = nudge_costs(bench, torch.full_like(mu, d))
        H = ((cp + cm - 2 * base) / d**2).cpu().numpy()
        pred = np.where(H > 0, np.sqrt(LAM / np.maximum(H, 1e-12)), SMAX).clip(SMIN, SMAX)
        rho = [spearman(sig_main[i], pred) for i in range(50)]
        med = np.median(sig_main, axis=0)
        free = (pred > SMIN * 1.01) & (pred < SMAX * 0.99)
        slope = (np.polyfit(np.log(pred[free]), np.log(med[free]), 1)[0]
                 if free.sum() > 10 else float("nan"))
        r_log = (float(np.corrcoef(np.log(pred[free]), np.log(med[free]))[0, 1])
                 if free.sum() > 10 else float("nan"))
        curv.append(dict(delta=d, frac_H_nonpos=float((H <= 0).mean()),
                         rho_sigma_pred=stats(rho), n_free=int(free.sum()),
                         loglog_slope=float(slope), loglog_r=r_log,
                         median_ratio=float(np.median(med[free] / pred[free]))))
        print(f"  delta {d}: H<=0 on {100 * (H <= 0).mean():.0f}% | rho(sigma, pred) "
              f"{np.mean(rho):+.3f}±{np.std(rho):.3f} | unclipped n={free.sum()} "
              f"log-log slope {slope:.2f} r {r_log:.2f} "
              f"median sigma/pred {np.median(med[free] / pred[free]):.2f}")
    out["curvature"] = curv

    with open("results_rev/analysis.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nsaved results_rev/analysis.json")


if __name__ == "__main__":
    main()
