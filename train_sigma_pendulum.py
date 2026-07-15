"""Stage A on Pendulum-v1: fix mu at the trained baseline, evolve per-weight sigma against
the negative ELBO with self-adaptive operators, test the importance-map hypothesis.

Same protocol as CartPole Stage A (random refs per evaluation, battery selection,
criticality on fixed conditions). lambda is swept: the cost floor here is ~17, so lambda
must be far smaller than on plants with cost ~2e4 (the KL-to-cost calibration lesson).

Run:  python train_sigma_pendulum.py [--lams 0.001,0.003,...]
Outputs: results_pd/sigma_lam<lam>_best.pt, results_pd/sigma_sweep.json,
         results_pd/sigma_vs_criticality.png
"""

import argparse
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

INPUTS = ("e", "theta", "theta_dot")     # the chosen no_ie set
HIDDEN = 8
SIGMA_MIN, SIGMA_MAX, SIGMA_PRIOR = 1e-3, 0.3, 1.0
K_TRAIN, K_RERANK, K_DIAG = 16, 8, 64
N_ISLANDS, POP, GENS, MIG = 9, 300, 800, 40
SEED = 0            # overridden by --seed


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = a.argsort().argsort().astype(float); rb = b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


PLAN = "self_adaptive"      # overridden by --plan; "default" = standard GA operators


def run_lambda(lam, mu, crit, scale, battery):
    n_w = mu.shape[0]
    E_bat = battery.shape[0]
    rng = np.random.default_rng(4242)
    mu_row = mu.reshape(1, -1)

    def evaluate(sigmas):
        genes = torch.cat([mu_row.expand(sigmas.shape[0], -1), sigmas], dim=1)
        refs = torch.stack([P.get_random_reference(rng=rng, device=DEV) for _ in range(2)])
        out = P.simulate_mc(genes, refs, INPUTS, scale, HIDDEN, K=K_TRAIN, episodes=2)
        return out["expected_cost"] + lam * B.kl_divergence(out["mu"], out["sigma"], SIGMA_PRIOR)

    space = get_space(n_w, SIGMA_MIN, SIGMA_MAX, device=DEV)
    torch.manual_seed(SEED)
    if PLAN == "default":
        plan, reset, tag = None, 151, "_stdops"     # standard GA operators (control run)
    else:
        plan, reset, tag = repro.self_adaptive_plan(tau=0.3), 0, ""
    ga = IslandGA(n_islands=N_ISLANDS, pop_size=POP, space=space, evaluate_fn=evaluate,
                  evaluate_detail_fn=None, num_generations=GENS, migration_interval=MIG,
                  reset_interval=reset, device=DEV,
                  log_dir=f"results_pd/sigma_lam{lam}_s{SEED}{tag}_log",
                  fitness_fn=f"E[cost]+{lam}*KL", plan=plan)
    t0 = time.time()
    ga.run()

    final_pop = torch.cat(ga.populations, dim=0)
    def battery_fit(sigmas):
        genes = torch.cat([mu_row.expand(sigmas.shape[0], -1), sigmas], dim=1)
        out = P.simulate_mc(genes, battery, INPUTS, scale, HIDDEN, K=K_RERANK,
                            episodes=E_bat, init_seed=555)
        return out["expected_cost"] + lam * B.kl_divergence(out["mu"], out["sigma"], SIGMA_PRIOR)
    fits = torch.cat([battery_fit(final_pop[i:i + 512])
                      for i in range(0, final_pop.shape[0], 512)])
    sigma = final_pop[fits.argmin()].clone()
    torch.save(sigma, f"results_pd/sigma_lam{lam}_s{SEED}{tag}_best.pt")

    genes = torch.cat([mu, sigma]).unsqueeze(0)
    out = P.simulate_mc(genes, battery, INPUTS, scale, HIDDEN, K=K_DIAG,
                        episodes=E_bat, init_seed=555)
    s = sigma.cpu().numpy()
    order = np.argsort(crit); dec = max(1, n_w // 10)
    rec = dict(lam=lam, spearman=spearman(s, crit),
               sigma_least=float(s[order[:dec]].mean()), sigma_most=float(s[order[-dec:]].mean()),
               sigma_mean=float(s.mean()), fell=out["fell_rate"].item(),
               E_cost=out["expected_cost"].item())
    print(f"  lam={lam:<7} rho={rec['spearman']:+.3f}  σ[least/most]="
          f"{rec['sigma_least']:.3f}/{rec['sigma_most']:.3f}  meanσ={rec['sigma_mean']:.3f}"
          f"  fell={rec['fell']:.3f}  E[cost]={rec['E_cost']:.1f}  ({time.time()-t0:.0f}s)")
    return rec, s


def main():
    global SEED, PLAN
    ap = argparse.ArgumentParser()
    ap.add_argument("--lams", default="0.001,0.003,0.01,0.03,0.1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plan", default="self_adaptive", choices=["self_adaptive", "default"])
    args = ap.parse_args()
    lams = [float(x) for x in args.lams.split(",")]
    SEED = args.seed
    PLAN = args.plan

    os.makedirs("results_pd", exist_ok=True)
    scale = P.scale_tensor(INPUTS, device=DEV)
    mu = torch.load("results_pd/baseline_no_ie_best.pt").to(DEV)
    battery = P.validation_battery(device=DEV)
    floor = P.simulate(mu.unsqueeze(0), battery, INPUTS, scale, HIDDEN,
                       episodes=battery.shape[0], init_seed=555)["cost"][0].item()
    crit = P.weight_criticality(mu, battery, INPUTS, scale, HIDDEN,
                                episodes=battery.shape[0], init_seed=555).cpu().numpy()
    print(f"mu battery cost (floor) = {floor:.1f} | criticality: min={crit.min():.3f} "
          f"p50={np.median(crit):.3f} max={crit.max():.1f} | sweeping lam={lams}\n")

    records, sigmas = [], {}
    for lam in lams:
        rec, s = run_lambda(lam, mu, crit, scale, battery)
        records.append(rec); sigmas[lam] = s

    with open(f"results_pd/sigma_sweep_s{SEED}.json", "w") as fh:
        json.dump({"floor": floor, "seed": SEED, "records": records}, fh, indent=2)

    print(f"\n{'='*72}\nSWEEP SUMMARY (floor {floor:.1f})")
    print(f"  {'lam':>7} {'Spearman':>9} {'σ_least':>8} {'σ_most':>8} {'σ_mean':>8} {'fell':>6} {'E[cost]':>9}")
    for r in records:
        print(f"  {r['lam']:>7} {r['spearman']:>+9.3f} {r['sigma_least']:>8.3f}"
              f" {r['sigma_most']:>8.3f} {r['sigma_mean']:>8.3f} {r['fell']:>6.3f} {r['E_cost']:>9.1f}")

    best = min(records, key=lambda r: r["spearman"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(crit + 1e-5, sigmas[best["lam"]], s=10, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("weight criticality (battery)")
    ax.set_ylabel("evolved sigma")
    ax.set_title(f"Pendulum importance map lam={best['lam']} (Spearman {best['spearman']:+.2f})")
    fig.tight_layout(); fig.savefig("results_pd/sigma_vs_criticality.png", dpi=120)
    print(f"\nstrongest map at lam={best['lam']}. Saved sweep json + png.")


if __name__ == "__main__":
    main()
