"""Pendulum calibration test: does the importance-map posterior's predictive uncertainty
rise when the plant drifts, and does the *structure* matter?

With mu fixed at the baseline we perturb native physics parameters (g, m, l, max_torque)
by factors 0.6..1.4 and measure, at each setting:
  degradation   = cost of the mean controller (battery refs, fixed initial states),
  uncertainty   = std of cost across K posterior weight samples,
for (a) the evolved importance-map sigma and (b) a uniform sigma of the same mean
(equal uncertainty budget, no structure).

Run:  python perturbation_pendulum.py [--lam 0.003] [--seed 0]
Outputs: results_pd/perturbation.json
"""

import argparse
import json
import os

import numpy as np
import torch

from ga_torch import pendulum as P
from ga_torch.simulation import setup_precision

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
setup_precision("high")
torch.manual_seed(0)          # deterministic MC sampling: reruns reproduce the paper numbers exactly

INPUTS = ("e", "theta", "theta_dot")
HIDDEN = 8
K = 64
PARAMS = ["g", "m", "l", "max_torque"]
SCALES = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = a.argsort().argsort().astype(float); rb = b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", default="0.003")
    ap.add_argument("--seed", default="0")
    args = ap.parse_args()

    scale = P.scale_tensor(INPUTS, device=DEV)
    mu = torch.load("results_pd/baseline_no_ie_best.pt").to(DEV)
    sig_map = torch.load(f"results_pd/sigma_lam{args.lam}_s{args.seed}_best.pt").to(DEV)
    sig_uni = torch.full_like(sig_map, sig_map.mean().item())
    battery = P.validation_battery(device=DEV)
    E = battery.shape[0]
    print(f"map sigma mean = {sig_map.mean():.4f} (uniform control = same)")

    genes_map = torch.cat([mu, sig_map]).unsqueeze(0)
    genes_uni = torch.cat([mu, sig_uni]).unsqueeze(0)

    results = {}
    for p in PARAMS:
        rec = {"scale": [], "degradation": [], "unc_map": [], "unc_uni": [],
               "fell_map": [], "fell_uni": []}
        for s in SCALES:
            ov = {p: P.PHYS[p] * s}
            deg = P.simulate(mu.unsqueeze(0), battery, INPUTS, scale, HIDDEN,
                             episodes=E, init_seed=555, plant_overrides=ov)["cost"][0].item()
            om = P.simulate_mc(genes_map, battery, INPUTS, scale, HIDDEN, K=K,
                               episodes=E, init_seed=555, plant_overrides=ov)
            ou = P.simulate_mc(genes_uni, battery, INPUTS, scale, HIDDEN, K=K,
                               episodes=E, init_seed=555, plant_overrides=ov)
            rec["scale"].append(s); rec["degradation"].append(deg)
            rec["unc_map"].append(om["cost_std"].item())
            rec["unc_uni"].append(ou["cost_std"].item())
            rec["fell_map"].append(om["fell_rate"].item())
            rec["fell_uni"].append(ou["fell_rate"].item())
        rec["rho_map"] = spearman(rec["unc_map"], rec["degradation"])
        rec["rho_uni"] = spearman(rec["unc_uni"], rec["degradation"])
        results[p] = rec
        print(f"  [{p:>10}] corr(uncertainty, degradation): map={rec['rho_map']:+.2f} "
              f"uniform={rec['rho_uni']:+.2f} | deg range {min(rec['degradation']):.0f}.."
              f"{max(rec['degradation']):.0f}")

    all_deg = [d for p in PARAMS for d in results[p]["degradation"]]
    all_map = [u for p in PARAMS for u in results[p]["unc_map"]]
    all_uni = [u for p in PARAMS for u in results[p]["unc_uni"]]
    pooled_map, pooled_uni = spearman(all_map, all_deg), spearman(all_uni, all_deg)
    print(f"\nPOOLED corr(uncertainty, degradation): importance-map={pooled_map:+.2f}  "
          f"uniform={pooled_uni:+.2f}")

    os.makedirs("results_pd", exist_ok=True)
    with open("results_pd/perturbation.json", "w") as fh:
        json.dump({"lam": args.lam, "seed": args.seed, "pooled_map": pooled_map,
                   "pooled_uni": pooled_uni, "results": results}, fh, indent=2)
    print("saved results_pd/perturbation.json")


if __name__ == "__main__":
    main()
