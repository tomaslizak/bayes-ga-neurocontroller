"""Seed-robustness study for the importance map (paper Sec. 4.2).

Two questions:
  (1) Is the map stable over many sigma-evolution seeds?  ->  50 seeds on the main baseline.
  (2) Is it specific to one trained controller?           ->  3 independent baselines
                                                              (different GA training seeds),
                                                              3 sigma-seeds each.

Reuses existing artifacts where present (sigma seeds 0-2 on the main baseline, the main
baseline itself) and trains only what is missing. Everything is selected on the fixed
validation battery, criticality is measured per baseline.

Run:  python robustness_seeds.py
Outputs: results_pd/robustness.json (+ baseline_b*.pt, sigma_b*_s*.pt for new runs)
"""

import json
import os
import time

import numpy as np
import torch

from ga_torch import IslandGA
from ga_torch import pendulum as P
from ga_torch import bayesian as B
from ga_torch import reproduction as repro
from ga_torch.population import get_space
from ga_torch.simulation import setup_precision

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
setup_precision("high")

INPUTS, HIDDEN = ("e", "theta", "theta_dot"), 8
LAM, SIGMA_PRIOR = 0.003, 1.0
SIGMA_MIN, SIGMA_MAX = 1e-3, 0.3
K_TRAIN, K_RERANK, K_DIAG = 16, 8, 64
N_ISLANDS, POP, MIG = 9, 300, 40
GENS_BASE, GENS_SIGMA = 1000, 800

SIGMA_SEEDS_MAIN = list(range(50))     # question (1)
BASELINE_SEEDS = [1, 2]                # question (2); baseline seed 0 = the main baseline
SIGMA_SEEDS_PER_BASELINE = [0, 1, 2]

battery = P.validation_battery(device=DEV)
E_BAT = battery.shape[0]
scale = P.scale_tensor(INPUTS, device=DEV)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = a.argsort().argsort().astype(float); rb = b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def train_baseline(bseed):
    """Train (or load) a deterministic baseline with GA training seed `bseed`."""
    path = ("results_pd/baseline_no_ie_best.pt" if bseed == 0
            else f"results_pd/baseline_b{bseed}.pt")
    if os.path.exists(path):
        return torch.load(path).to(DEV), path
    n_genes = P.gene_count(len(INPUTS), HIDDEN)
    rng = np.random.default_rng(12345 + bseed)
    def evaluate(genes):
        refs = torch.stack([P.get_random_reference(rng=rng, device=DEV) for _ in range(3)])
        return P.simulate(genes, refs, INPUTS, scale, HIDDEN, 3)["cost"]
    torch.manual_seed(bseed)
    ga = IslandGA(n_islands=N_ISLANDS, pop_size=POP,
                  space=get_space(n_genes, -1.0, 1.0, device=DEV),
                  evaluate_fn=evaluate, evaluate_detail_fn=None,
                  num_generations=GENS_BASE, migration_interval=MIG, reset_interval=201,
                  device=DEV, log_dir=f"results_pd/baseline_b{bseed}_log",
                  fitness_fn="baseline")
    ga.run()
    final = torch.cat(ga.populations, dim=0)
    costs = torch.cat([P.simulate(final[i:i+1024], battery, INPUTS, scale, HIDDEN,
                                  episodes=E_BAT, init_seed=555)["cost"]
                       for i in range(0, final.shape[0], 1024)])
    best = final[costs.argmin()].clone()
    torch.save(best, path)
    return best, path


def evolve_sigma(mu, sseed, tag):
    """Evolve (or load) sigma for baseline `mu` with sigma-seed `sseed`."""
    path = f"results_pd/sigma_{tag}_best.pt"
    legacy = f"results_pd/sigma_lam{LAM}_s{sseed}_best.pt"
    if os.path.exists(path):
        return torch.load(path).to(DEV)
    if tag.startswith("b0_") and os.path.exists(legacy):
        return torch.load(legacy).to(DEV)
    n_w = mu.shape[0]
    rng = np.random.default_rng(4242 + sseed)
    mu_row = mu.reshape(1, -1)
    def evaluate(sigmas):
        genes = torch.cat([mu_row.expand(sigmas.shape[0], -1), sigmas], dim=1)
        refs = torch.stack([P.get_random_reference(rng=rng, device=DEV) for _ in range(2)])
        out = P.simulate_mc(genes, refs, INPUTS, scale, HIDDEN, K=K_TRAIN, episodes=2)
        return out["expected_cost"] + LAM * B.kl_divergence(out["mu"], out["sigma"], SIGMA_PRIOR)
    torch.manual_seed(sseed)
    ga = IslandGA(n_islands=N_ISLANDS, pop_size=POP,
                  space=get_space(n_w, SIGMA_MIN, SIGMA_MAX, device=DEV),
                  evaluate_fn=evaluate, evaluate_detail_fn=None,
                  num_generations=GENS_SIGMA, migration_interval=MIG, reset_interval=0,
                  device=DEV, log_dir=f"results_pd/sigma_{tag}_log",
                  fitness_fn=f"E[cost]+{LAM}*KL",
                  plan=repro.self_adaptive_plan(tau=0.3))
    ga.run()
    final = torch.cat(ga.populations, dim=0)
    def bat_fit(sigmas):
        genes = torch.cat([mu_row.expand(sigmas.shape[0], -1), sigmas], dim=1)
        out = P.simulate_mc(genes, battery, INPUTS, scale, HIDDEN, K=K_RERANK,
                            episodes=E_BAT, init_seed=555)
        return out["expected_cost"] + LAM * B.kl_divergence(out["mu"], out["sigma"], SIGMA_PRIOR)
    fits = torch.cat([bat_fit(final[i:i+512]) for i in range(0, final.shape[0], 512)])
    sigma = final[fits.argmin()].clone()
    torch.save(sigma, path)
    return sigma


def diagnose(mu, sigma, crit):
    g = torch.cat([mu, sigma]).unsqueeze(0)
    out = P.simulate_mc(g, battery, INPUTS, scale, HIDDEN, K=K_DIAG,
                        episodes=E_BAT, init_seed=555)
    s = sigma.cpu().numpy()
    return dict(rho=spearman(s, crit), fell=out["fell_rate"].item(),
                E_cost=out["expected_cost"].item(), sigma_mean=float(s.mean()))


def main():
    t_all = time.time()
    results = {"main_baseline_50seeds": [], "multi_baseline": []}

    # ---- (1) 50 sigma-seeds on the main baseline ----
    mu0, _ = train_baseline(0)
    crit0 = P.weight_criticality(mu0, battery, INPUTS, scale, HIDDEN,
                                 episodes=E_BAT, init_seed=555).cpu().numpy()
    floor0 = P.simulate(mu0.unsqueeze(0), battery, INPUTS, scale, HIDDEN,
                        episodes=E_BAT, init_seed=555)["cost"][0].item()
    print(f"(1) main baseline: floor={floor0:.1f}; sigma seeds {SIGMA_SEEDS_MAIN}")
    for s in SIGMA_SEEDS_MAIN:
        t0 = time.time()
        sig = evolve_sigma(mu0, s, f"b0_s{s}")
        rec = diagnose(mu0, sig, crit0); rec["seed"] = s
        results["main_baseline_50seeds"].append(rec)
        print(f"    seed {s}: rho={rec['rho']:+.3f}  fell={rec['fell']:.3f}  "
              f"E[cost]={rec['E_cost']:.1f}  ({time.time()-t0:.0f}s)")

    # ---- (2) independent baselines ----
    print(f"(2) independent baselines {BASELINE_SEEDS}, sigma seeds {SIGMA_SEEDS_PER_BASELINE}")
    for b in BASELINE_SEEDS:
        t0 = time.time()
        mub, path = train_baseline(b)
        floorb = P.simulate(mub.unsqueeze(0), battery, INPUTS, scale, HIDDEN,
                            episodes=E_BAT, init_seed=555)["cost"][0].item()
        critb = P.weight_criticality(mub, battery, INPUTS, scale, HIDDEN,
                                     episodes=E_BAT, init_seed=555).cpu().numpy()
        print(f"  baseline b{b}: floor={floorb:.1f} ({time.time()-t0:.0f}s)")
        for s in SIGMA_SEEDS_PER_BASELINE:
            t0 = time.time()
            sig = evolve_sigma(mub, s, f"b{b}_s{s}")
            rec = diagnose(mub, sig, critb)
            rec.update(baseline=b, seed=s, floor=floorb)
            results["multi_baseline"].append(rec)
            print(f"    b{b} s{s}: rho={rec['rho']:+.3f}  fell={rec['fell']:.3f}  "
                  f"E[cost]={rec['E_cost']:.1f}  ({time.time()-t0:.0f}s)")

    # ---- summary ----
    r10 = [x["rho"] for x in results["main_baseline_50seeds"]]
    print(f"\n{'='*60}\n(1) 50 seeds, main baseline:  rho = {np.mean(r10):+.3f} "
          f"+- {np.std(r10):.3f}   range [{min(r10):+.3f}, {max(r10):+.3f}]")
    for b in BASELINE_SEEDS:
        rb = [x["rho"] for x in results["multi_baseline"] if x["baseline"] == b]
        print(f"(2) baseline b{b} (3 seeds):     rho = {np.mean(rb):+.3f} +- {np.std(rb):.3f}")
    rall = r10 + [x["rho"] for x in results["multi_baseline"]]
    print(f"    ALL {len(rall)} runs pooled:       rho = {np.mean(rall):+.3f} +- {np.std(rall):.3f}")
    with open("results_pd/robustness.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"saved results_pd/robustness.json  (total {time.time()-t_all:.0f}s)")


if __name__ == "__main__":
    main()
