"""Capability-boundary sweep (paper Sec. 4.5, Fig 5): command a constant held set-point of
increasing magnitude and record the mean controller's cost and the posterior's predictive
uncertainty at each level. The uncertainty erupts at 0.30 rad, one step before the mean
controller fails at 0.32 rad.

Run:  python hold_sweep.py
Output: results_pd/hold_sweep.json (then `python make_figures.py` redraws Fig 5)
"""

import json

import torch

from ga_torch import pendulum as P
from ga_torch.simulation import setup_precision

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
setup_precision("high")
torch.manual_seed(0)          # deterministic MC sampling: reruns reproduce the paper numbers exactly

INPUTS, HIDDEN = ("e", "theta", "theta_dot"), 8
LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38]

scale = P.scale_tensor(INPUTS, device=DEV)
mu = torch.load("results_pd/baseline_no_ie_best.pt").to(DEV)
sig = torch.load("results_pd/sigma_lam0.003_s0_best.pt").to(DEV)
genes = torch.cat([mu, sig]).unsqueeze(0)

out = {"level": LEVELS, "unc": [], "fell": [], "det_cost": []}
for lv in LEVELS:
    ref = torch.full((P.NSTEPS,), float(lv), device=DEV)
    ref[:40] = 0.0                                   # 2 s settle at upright first
    mc = P.simulate_mc(genes, ref, INPUTS, scale, HIDDEN, K=64, episodes=8, init_seed=99)
    det = P.simulate(mu.unsqueeze(0), ref, INPUTS, scale, HIDDEN, episodes=8, init_seed=99)
    out["unc"].append(mc["cost_std"].item())
    out["fell"].append(mc["fell_rate"].item())
    out["det_cost"].append(det["cost"][0].item())
    print(f"hold {lv:.2f} rad: mean-controller cost {out['det_cost'][-1]:8.1f}   "
          f"uncertainty {out['unc'][-1]:8.1f}   sample fall rate {out['fell'][-1]:.3f}")

with open("results_pd/hold_sweep.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("saved results_pd/hold_sweep.json")
