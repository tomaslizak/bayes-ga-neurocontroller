"""Revision experiments on Pendulum-v1 with the unchanged method (work package A).

One function, `run_one`, evolves sigma for a given configuration and scores the result with
a common high-K scorer. The default configuration is the one used in the paper
(lambda 0.003, sigma_p 1, sigma in [1e-3, 0.3], K 16, 9 islands x 300, 800 generations,
self-adaptive operators); every study changes one setting at a time:

  A1  lambda sweep          A2  standard operators      A3  prior width sigma_p
  A4  sigma cap             A5  Monte-Carlo samples K   A6  population size
  A7  number of generations

The queue is resumable: a run whose JSON record exists is skipped.

Run:   python rev_pendulum.py            # whole queue
       python rev_pendulum.py --smoke    # 3-generation self-test
       python rev_pendulum.py --summary  # table from finished runs
Outputs: results_rev/runs/<tag>.json, results_rev/sigma/<tag>.pt, results_rev/<tag>_log/
"""

import argparse
import glob
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
OUT = "results_rev"
MIG = 40
K_RERANK, K_DIAG = 8, 2048
EPISODES_TRAIN = 2

DEFAULT = dict(lam=0.003, sigma_prior=1.0, sigma_min=1e-3, sigma_max=0.3, K=16,
               n_islands=9, pop=300, gens=800, plan="self_adaptive")


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = a.argsort().argsort().astype(float); rb = b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


class Bench:
    """The fixed pieces shared by all runs: baseline mu, battery, criticality, floor."""

    def __init__(self, mu_path="results_pd/baseline_no_ie_best.pt", hidden=HIDDEN,
                 arch="gru", inputs=INPUTS, env=P, scale_map=None):
        # env: the plant module (pendulum, or another module with the same functions)
        self.hidden, self.arch, self.inputs, self.env = hidden, arch, inputs, env
        self.scale = env.scale_tensor(inputs, scale_map, device=DEV)
        self.mu = torch.load(mu_path).to(DEV)
        self.battery = env.validation_battery(device=DEV)
        self.E = self.battery.shape[0]
        self.floor = env.simulate(self.mu.unsqueeze(0), self.battery, inputs, self.scale,
                                hidden, episodes=self.E, init_seed=555,
                                arch=arch)["cost"][0].item()
        self.crit = env.weight_criticality(self.mu, self.battery, inputs, self.scale, hidden,
                                         episodes=self.E, init_seed=555,
                                         arch=arch).cpu().numpy()

    def score(self, sigma, sigma_prior=1.0, K=K_DIAG, seed=999, plant_overrides=None):
        """Common scorer: K posterior samples on the frozen battery, fixed noise seed."""
        gen = torch.Generator(device=DEV).manual_seed(seed)
        genes = torch.cat([self.mu, sigma]).unsqueeze(0)
        out = self.env.simulate_mc(genes, self.battery, self.inputs, self.scale, self.hidden,
                                   K=K, episodes=self.E, init_seed=555, generator=gen,
                                   arch=self.arch, plant_overrides=plant_overrides)
        s = sigma.cpu().numpy()
        order = np.argsort(self.crit); dec = max(1, len(s) // 10)
        return dict(
            rho=spearman(s, self.crit),
            sigma_mean=float(s.mean()),
            sigma_least=float(s[order[:dec]].mean()),      # least-critical decile
            sigma_most=float(s[order[-dec:]].mean()),      # most-critical decile
            fell=out["fell_rate"].item(),
            E_cost=out["expected_cost"].item(),
            cost_std=out["cost_std"].item(),
            kl=B.kl_divergence(self.mu.unsqueeze(0), sigma.unsqueeze(0), sigma_prior).item(),
        )


def run_one(bench, tag, seed, out=OUT, **overrides):
    cfg = dict(DEFAULT); cfg.update(overrides)
    rec_path = f"{out}/runs/{tag}.json"
    if os.path.exists(rec_path):
        return json.load(open(rec_path))
    os.makedirs(f"{out}/runs", exist_ok=True); os.makedirs(f"{out}/sigma", exist_ok=True)

    lam, sp, K = cfg["lam"], cfg["sigma_prior"], cfg["K"]
    mu_row = bench.mu.reshape(1, -1)
    n_w = bench.mu.shape[0]
    rng = np.random.default_rng(4242 + seed)

    def evaluate(sigmas):
        genes = torch.cat([mu_row.expand(sigmas.shape[0], -1), sigmas], dim=1)
        refs = torch.stack([bench.env.get_random_reference(rng=rng, device=DEV)
                            for _ in range(EPISODES_TRAIN)])
        o = bench.env.simulate_mc(genes, refs, bench.inputs, bench.scale, bench.hidden, K=K,
                                  episodes=EPISODES_TRAIN, arch=bench.arch)
        return o["expected_cost"] + lam * B.kl_divergence(o["mu"], o["sigma"], sp)

    if cfg["plan"] == "default":            # standard GA operators (control)
        plan, reset = None, 151
    else:
        plan, reset = repro.self_adaptive_plan(tau=0.3), 0

    torch.manual_seed(seed)
    ga = IslandGA(n_islands=cfg["n_islands"], pop_size=cfg["pop"],
                  space=get_space(n_w, cfg["sigma_min"], cfg["sigma_max"], device=DEV),
                  evaluate_fn=evaluate, evaluate_detail_fn=None,
                  num_generations=cfg["gens"], migration_interval=MIG, reset_interval=reset,
                  device=DEV, log_dir=f"{out}/{tag}_log",
                  fitness_fn=f"E[cost]+{lam}*KL", plan=plan)
    t0 = time.time()
    ga.run()
    t_ga = time.time() - t0

    final = torch.cat(ga.populations, dim=0)

    def bat_fit(sigmas):
        genes = torch.cat([mu_row.expand(sigmas.shape[0], -1), sigmas], dim=1)
        o = bench.env.simulate_mc(genes, bench.battery, bench.inputs, bench.scale,
                                  bench.hidden, K=K_RERANK, episodes=bench.E, init_seed=555,
                                  arch=bench.arch)
        return o["expected_cost"] + lam * B.kl_divergence(o["mu"], o["sigma"], sp)

    fits = torch.cat([bat_fit(final[i:i + 512]) for i in range(0, final.shape[0], 512)])
    sigma = final[fits.argmin()].clone()
    torch.save(sigma.cpu(), f"{out}/sigma/{tag}.pt")

    s = sigma.cpu().numpy()
    n_ind = cfg["n_islands"] * cfg["pop"]
    rec = dict(tag=tag, seed=seed, cfg=cfg, floor=bench.floor, hidden=bench.hidden,
               arch=bench.arch, inputs=list(bench.inputs), n_weights=n_w)
    rec.update(bench.score(sigma, sp))
    rec.update(
        frac_at_cap=float((s >= 0.98 * cfg["sigma_max"]).mean()),
        frac_at_floor=float((s <= 1.5 * cfg["sigma_min"]).mean()),
        t_ga_s=t_ga, t_total_s=time.time() - t0,
        rollouts=n_ind * K * EPISODES_TRAIN * cfg["gens"],
        plant_steps=n_ind * K * EPISODES_TRAIN * cfg["gens"] * bench.env.NSTEPS,
    )
    with open(rec_path, "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"### {tag}: rho={rec['rho']:+.3f} fell={rec['fell']:.4f} "
          f"E[J]={rec['E_cost']:.2f} mean_sigma={rec['sigma_mean']:.3f} "
          f"({t_ga:.0f}s)", flush=True)
    return rec


LAMS = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]


def build_queue():
    """(tag, seed, overrides), most important first."""
    q = []
    for s in range(5):                                   # A1 lambda sweep
        for lam in LAMS:
            q.append((f"lam{lam:g}_s{s}", s, dict(lam=lam)))
    for s in range(5):                                   # A2 standard operators
        q.append((f"stdops_s{s}", s, dict(plan="default")))
    for s in range(3):                                   # A3 prior width
        for sp in [0.1, 0.3, 3.0, 10.0]:
            q.append((f"sp{sp:g}_s{s}", s, dict(sigma_prior=sp)))
    for s in range(3):                                   # A5 Monte-Carlo samples
        for K in [2, 4, 8, 32]:
            q.append((f"K{K}_s{s}", s, dict(K=K)))
    for s in range(3):                                   # A6 population
        for ni, pop in [(9, 100), (3, 100), (1, 100)]:
            q.append((f"pop{ni}x{pop}_s{s}", s, dict(n_islands=ni, pop=pop)))
    for s in range(3):                                   # A7 generations
        for g in [200, 400]:
            q.append((f"gens{g}_s{s}", s, dict(gens=g)))
    for s in range(3):                                   # A4 sigma cap
        for smax in [0.1, 1.0]:
            q.append((f"smax{smax:g}_s{s}", s, dict(sigma_max=smax)))
    return q


def summary(out=OUT):
    recs = [json.load(open(p)) for p in sorted(glob.glob(f"{out}/runs/*.json"))]
    groups = {}
    for r in recs:
        groups.setdefault(r["tag"].rsplit("_s", 1)[0], []).append(r)
    cols = ["rho", "fell", "E_cost", "cost_std", "sigma_mean", "sigma_most", "sigma_least",
            "frac_at_cap", "t_ga_s"]
    print(f"{'group':<14}{'n':>3} " + " ".join(f"{c:>17}" for c in cols))
    for g, rs in groups.items():
        cells = []
        for c in cols:
            v = np.array([r[c] for r in rs])
            cells.append(f"{v.mean():>9.4g}±{v.std():<7.2g}")
        print(f"{g:<14}{len(rs):>3} " + " ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--only", default="", help="run only tags starting with this prefix")
    args = ap.parse_args()
    if args.summary:
        return summary()
    bench = Bench()
    print(f"floor={bench.floor:.3f} | criticality p50={np.median(bench.crit):.3f} "
          f"max={bench.crit.max():.1f}", flush=True)
    if args.smoke:
        for tag, ov in [("smoke_default", {}), ("smoke_one_island", dict(n_islands=1, pop=100)),
                        ("smoke_stdops", dict(plan="default")), ("smoke_K32", dict(K=32))]:
            r = run_one(bench, tag, 0, out=f"{OUT}/smoke", gens=3, **ov)
            print({k: r[k] for k in ("rho", "fell", "E_cost", "t_ga_s", "rollouts")})
        return
    queue = [q for q in build_queue() if q[0].startswith(args.only)]
    t0 = time.time()
    for i, (tag, seed, ov) in enumerate(queue):
        print(f"\n===== [{i + 1}/{len(queue)}] {tag}  (elapsed {time.time() - t0:.0f}s) =====",
              flush=True)
        run_one(bench, tag, seed, **ov)
    print(f"queue finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
