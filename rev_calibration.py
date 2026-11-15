"""Calibration figures re-evaluated with many posterior samples (work package A9).

The submitted figures 4 and 5 use K = 64 posterior samples and report the standard
deviation of the cost. With rare falling samples (about 0.1 % at the nominal plant) that
estimate depends strongly on whether one of the 64 samples happens to fall. This script
repeats both tests with K = 2048 and records, per setting,
  std        standard deviation of the cost over samples (the paper's uncertainty),
  spread     robust spread, interquartile range / 1.349 (equals the std for a Gaussian),
  any_fall   share of posterior samples with at least one fallen episode,
  mean       expected cost,
for the evolved sigma and for the uniform sigma of the same mean. Both benchmarks.

Run:   python rev_calibration.py
Output: results_rev/calibration_K2048.json
"""

import json

import numpy as np
import torch

from ga_torch import pendulum as P
from ga_torch import mountaincar_track as T
from rev_pendulum import Bench, DEV, OUT, spearman

K = 2048
SCALES = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]


@torch.no_grad()
def sample_stats(b, sigma, ref, episodes, init_seed, ov=None, seed=2024):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    W = b.mu.unsqueeze(0) + sigma.unsqueeze(0) * torch.randn(K, b.mu.shape[0], device=DEV,
                                                             generator=gen)
    o = b.env.simulate(W, ref, b.inputs, b.scale, b.hidden, episodes, init_seed=init_seed,
                       plant_overrides=ov, arch=b.arch)
    c = o["cost"]
    q = torch.quantile(c, torch.tensor([0.25, 0.5, 0.75, 0.95], device=DEV))
    return dict(std=c.std().item(), spread=((q[2] - q[0]) / 1.349).item(),
                median=q[1].item(), p95=q[3].item(), mean=c.mean().item(),
                any_fall=(o["fell_rate"] > 0).float().mean().item(),
                fell=o["fell_rate"].mean().item())


def run(name, b, sig_map, params, hold_levels, hold_ref):
    sig_uni = torch.full_like(sig_map, sig_map.mean().item())
    E = b.E
    res = {"perturbation": {}, "hold": []}
    print(f"\n===== {name}: sigma mean {sig_map.mean():.4f} =====")
    for p in params:
        rows = []
        for s in SCALES:
            ov = {p: b.env.PHYS[p] * s}
            deg = b.env.simulate(b.mu.unsqueeze(0), b.battery, b.inputs, b.scale, b.hidden, E,
                                 init_seed=555, plant_overrides=ov, arch=b.arch)["cost"][0].item()
            rows.append(dict(scale=s, degradation=deg,
                             map=sample_stats(b, sig_map, b.battery, E, 555, ov),
                             uni=sample_stats(b, sig_uni, b.battery, E, 555, ov)))
        res["perturbation"][p] = rows
        print(f"[{p}]  factor: mean-ctrl cost | map std / spread / any-fall | uniform std / spread / any-fall")
        for r in rows:
            m, u = r["map"], r["uni"]
            print(f"   x{r['scale']:<4} {r['degradation']:8.1f} | {m['std']:8.2f} {m['spread']:7.3f} "
                  f"{m['any_fall']:6.3f} | {u['std']:8.2f} {u['spread']:7.3f} {u['any_fall']:6.3f}")
    deg = [r["degradation"] for p in params for r in res["perturbation"][p]]
    for stat in ("std", "spread", "any_fall"):
        cm = spearman([r["map"][stat] for p in params for r in res["perturbation"][p]], deg)
        cu = spearman([r["uni"][stat] for p in params for r in res["perturbation"][p]], deg)
        res[f"pooled_corr_{stat}"] = dict(map=cm, uni=cu)
        print(f"pooled rank correlation of {stat:<8} with degradation: map {cm:+.2f}  uniform {cu:+.2f}")
    print("hold sweep  level: mean-ctrl cost | map std / spread / any-fall | uniform std / spread")
    for lv in hold_levels:
        ref = hold_ref(lv)
        det = b.env.simulate(b.mu.unsqueeze(0), ref, b.inputs, b.scale, b.hidden, 8,
                             init_seed=99, arch=b.arch)["cost"][0].item()
        m, u = sample_stats(b, sig_map, ref, 8, 99), sample_stats(b, sig_uni, ref, 8, 99)
        res["hold"].append(dict(level=lv, det_cost=det, map=m, uni=u))
        print(f"   {lv:<5} {det:8.1f} | {m['std']:8.2f} {m['spread']:7.3f} {m['any_fall']:6.3f} | "
              f"{u['std']:8.2f} {u['spread']:7.3f}")
    return res


def main():
    out = {"K": K}

    def pend_ref(lv):
        r = torch.full((P.NSTEPS,), float(lv), device=DEV); r[:40] = 0.0; return r

    def mc_ref(d):
        r = torch.full((T.NSTEPS,), T.X_BOTTOM + d, device=DEV); r[:T.SETTLE] = T.X_BOTTOM
        return r

    b = Bench()
    pend_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38]
    for tag, path in (("pendulum_seed0_lam0.003", "results_pd/sigma_lam0.003_s0_best.pt"),
                      ("pendulum_seed0_lam0.001", f"{OUT}/sigma/lam0.001_s0.pt")):
        sig = torch.load(path).to(DEV)
        out[tag] = run(tag, b, sig, ["g", "m", "l", "max_torque"], pend_levels, pend_ref)

    info = json.load(open(f"{OUT}/mct_baseline.json"))
    bm = Bench(f"{OUT}/mct_baseline.pt", inputs=("e", "x", "v"), env=T, scale_map=info["scale"])
    sig = torch.load(f"{OUT}/sigma/mct_lam0.003_s0.pt").to(DEV)
    mc_levels = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.17, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.26]
    out["mountaincar_seed0_lam0.003"] = run("mountaincar_seed0_lam0.003", bm, sig,
                                            ["power", "gravity"], mc_levels, mc_ref)
    with open(f"{OUT}/calibration_K2048.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nsaved results_rev/calibration_K2048.json")


if __name__ == "__main__":
    main()
