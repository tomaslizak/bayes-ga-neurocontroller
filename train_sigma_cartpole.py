"""Stage A on CartPole: fix mu at the trained baseline, evolve per-weight sigma against the
negative ELBO with self-adaptive operators, and test the importance-map hypothesis
(sigma small on critical weights, large on irrelevant ones).

Mirrors train_sigma_only.py (ODE plant) with the CartPole-specific protocol:
  * the data term uses FRESH random references each evaluation (same anti-memorisation
    protocol as baseline training),
  * final sigma is selected by re-ranking the last population on the fixed validation
    battery (fitnesses are not comparable across generations under random references),
  * criticality is measured on the battery with fixed initial states.

Run:  python train_sigma_cartpole.py [--inputs all6|no_x|no_ie]
Outputs: results_cp/sigma_lam<lam>_best.pt, results_cp/sigma_stage_a.json,
         results_cp/sigma_vs_criticality.png
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from ga_torch import IslandGA
from ga_torch import cartpole as CP
from ga_torch import bayesian as B
from ga_torch import reproduction as repro
from ga_torch.simulation import setup_precision

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
setup_precision("high")

INPUT_SETS = {
    "all6":  ("e", "ie", "x", "x_dot", "theta", "theta_dot"),
    "no_x":  ("e", "ie", "x_dot", "theta", "theta_dot"),
    "no_ie": ("e", "x", "x_dot", "theta", "theta_dot"),
}
HIDDEN = 8
# sigma_max is a stability cap and is task-dependent: pole balance is much sharper to
# JOINT weight noise than the ODE loop (individually-innocent weights at sigma=0.3
# jointly topple the pole ~20% of the time, at any lambda), so CartPole gets a tighter
# ceiling than the ODE plant's 0.3.
SIGMA_MIN, SIGMA_MAX, SIGMA_PRIOR = 1e-3, 0.1, 1.0
K_TRAIN, K_RERANK, K_DIAG = 16, 8, 64
N_ISLANDS, POP, GENS, MIG = 9, 300, 800, 40
SEED = 0


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = a.argsort().argsort().astype(float); rb = b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="all6", choices=list(INPUT_SETS))
    ap.add_argument("--lam", type=float, default=1.0)
    args = ap.parse_args()

    inputs = INPUT_SETS[args.inputs]
    lam = args.lam
    os.makedirs("results_cp", exist_ok=True)
    scale = CP.scale_tensor(inputs, device=DEV)
    mu = torch.load(f"results_cp/baseline_{args.inputs}_best.pt").to(DEV)
    n_w = mu.shape[0]
    battery = CP.validation_battery(device=DEV)
    E_bat = battery.shape[0]

    base_cost = CP.simulate(mu.unsqueeze(0), battery, inputs, scale, HIDDEN,
                            episodes=E_bat, init_seed=555)["cost"][0].item()
    print(f"inputs={args.inputs} ({len(inputs)} in, {n_w} weights) | lam={lam} | "
          f"mu battery cost (floor) = {base_cost:.1f}")

    # criticality of the fixed mu, on the battery with fixed initial states
    crit = CP.weight_criticality(mu, battery, inputs, scale, HIDDEN,
                                 episodes=E_bat, init_seed=555).cpu().numpy()
    print(f"criticality: min={crit.min():.2f} p50={np.median(crit):.2f} max={crit.max():.2f}")

    # --- sigma-only evolution (self-adaptive operators, fresh random refs each eval) ---
    rng = np.random.default_rng(4242)
    mu_row = mu.reshape(1, -1)

    def evaluate(sigmas):
        genes = torch.cat([mu_row.expand(sigmas.shape[0], -1), sigmas], dim=1)
        ref = torch.stack([CP.get_random_reference(rng=rng, device=DEV) for _ in range(2)])
        out = CP.simulate_mc(genes, ref, inputs, scale, HIDDEN, K=K_TRAIN, episodes=2)
        kl = B.kl_divergence(out["mu"], out["sigma"], SIGMA_PRIOR)
        return out["expected_cost"] + lam * kl

    from ga_torch.population import get_space
    space = get_space(n_w, SIGMA_MIN, SIGMA_MAX, device=DEV)

    torch.manual_seed(SEED)
    ga = IslandGA(n_islands=N_ISLANDS, pop_size=POP, space=space, evaluate_fn=evaluate,
                  evaluate_detail_fn=None, num_generations=GENS, migration_interval=MIG,
                  reset_interval=0, device=DEV,
                  log_dir=f"results_cp/sigma_lam{lam}_log",
                  fitness_fn=f"E[cost]+{lam}*KL (sigma-only, cartpole)",
                  plan=repro.self_adaptive_plan(tau=0.3))
    t0 = time.time()
    ga.run()
    print(f"evolved in {time.time()-t0:.0f}s")

    # --- select on the fixed battery (chunked), then diagnose at high K ---
    final_pop = torch.cat(ga.populations, dim=0)
    def battery_fitness(sigmas):
        genes = torch.cat([mu_row.expand(sigmas.shape[0], -1), sigmas], dim=1)
        out = CP.simulate_mc(genes, battery, inputs, scale, HIDDEN, K=K_RERANK,
                             episodes=E_bat, init_seed=555)
        kl = B.kl_divergence(out["mu"], out["sigma"], SIGMA_PRIOR)
        return out["expected_cost"] + lam * kl

    fits = torch.cat([battery_fitness(final_pop[i:i + 128])
                      for i in range(0, final_pop.shape[0], 128)])
    sigma = final_pop[fits.argmin()].clone()
    torch.save(sigma, f"results_cp/sigma_lam{lam}_best.pt")

    genes = torch.cat([mu, sigma]).unsqueeze(0)
    out = CP.simulate_mc(genes, battery, inputs, scale, HIDDEN, K=K_DIAG,
                         episodes=E_bat, init_seed=555)
    s = sigma.cpu().numpy()
    order = np.argsort(crit); dec = max(1, n_w // 10)
    rho = spearman(s, crit)
    sig_least, sig_most = s[order[:dec]].mean(), s[order[-dec:]].mean()

    print(f"\nE[cost] @ best sigma = {out['expected_cost'].item():.1f} (floor {base_cost:.1f})"
          f" | blown_rate = {out['blown_rate'].item():.3f}")
    print(f"sigma: min={s.min():.3f} p50={np.median(s):.3f} mean={s.mean():.3f} max={s.max():.3f}")
    print(f"IMPORTANCE MAP: Spearman(sigma, criticality) = {rho:+.3f}")
    print(f"  mean sigma LEAST-critical 10% = {sig_least:.3f} | MOST-critical 10% = {sig_most:.3f}")

    with open("results_cp/sigma_stage_a.json", "w") as fh:
        json.dump(dict(inputs=args.inputs, lam=lam, floor=base_cost,
                       E_cost=out["expected_cost"].item(), blown=out["blown_rate"].item(),
                       spearman=rho, sigma_least=float(sig_least), sigma_most=float(sig_most),
                       sigma_mean=float(s.mean())), fh, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(crit + 1e-4, s, s=10, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("weight criticality (battery)")
    ax.set_ylabel("evolved sigma")
    ax.set_title(f"CartPole importance map lam={lam} (Spearman {rho:+.2f})")
    fig.tight_layout(); fig.savefig("results_cp/sigma_vs_criticality.png", dpi=120)
    print("saved results_cp/sigma_lam%s_best.pt, sigma_stage_a.json, sigma_vs_criticality.png" % lam)


if __name__ == "__main__":
    main()
