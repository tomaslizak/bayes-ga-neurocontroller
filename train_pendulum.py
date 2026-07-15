"""Pendulum pipeline, step 1: train the deterministic GRU angle-tracking baseline on
(validated) Pendulum-v1 dynamics, verify held-out tracking, measure input ranges (1/p99),
and compare input sets. Protocol identical to the CartPole port: random references per
evaluation, fixed validation battery for selection, canonical reference held out.

Run:  python train_pendulum.py
Outputs: results_pd/baseline_<name>_best.pt, results_pd/baseline_compare.json
"""

import json
import os
import time

import numpy as np
import torch

from ga_torch import IslandGA
from ga_torch import pendulum as P
from ga_torch.population import get_space
from ga_torch.simulation import setup_precision

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
setup_precision("high")

CONFIGS = {
    "all4":  ("e", "ie", "theta", "theta_dot"),
    "no_ie": ("e", "theta", "theta_dot"),
}
HIDDEN = 8
SEED = 0
EPISODES_TRAIN = 3
N_ISLANDS, POP, GENS, MIG, RESET = 9, 300, 1000, 40, 201

SEGMENTS = [(2, 7, 0.25), (7, 12, -0.20), (12, 17, 0.12), (17, 22, 0.0)]


def evaluate_builder(inputs, scale):
    rng = np.random.default_rng(12345)
    def evaluate(genes):
        refs = torch.stack([P.get_random_reference(rng=rng, device=genes.device)
                            for _ in range(EPISODES_TRAIN)])
        return P.simulate(genes, refs, inputs, scale, HIDDEN, EPISODES_TRAIN)["cost"]
    return evaluate


def select_on_battery(final_pop, inputs, scale, chunk=1024):
    battery = P.validation_battery(device=final_pop.device)
    E = battery.shape[0]
    costs = torch.cat([P.simulate(final_pop[i:i + chunk], battery, inputs, scale,
                                  HIDDEN, episodes=E, init_seed=555)["cost"]
                       for i in range(0, final_pop.shape[0], chunk)])
    idx = costs.argmin()
    return final_pop[idx].clone(), costs[idx].item()


def tracking_report(best, ref, inputs, scale):
    out = P.simulate(best.unsqueeze(0), ref, inputs, scale, HIDDEN,
                     episodes=4, init_seed=99, return_traj=True)
    th = out["theta"].reshape(4, -1).mean(dim=0)
    print("    segment(s)  setpoint  achieved      err")
    ok, worst = True, 0.0
    for (a, b, sp) in SEGMENTS:
        i0, i1 = int((b - 1.5) / P.DT), int(b / P.DT)
        ach = th[i0:i1].mean().item()
        err = ach - sp
        worst = max(worst, abs(err))
        flag = "" if abs(err) < 0.05 else "  <-- off"
        ok = ok and not flag
        print(f"    [{a:>3},{b:>3}) {sp:8.2f} {ach:9.3f} {err:8.3f}{flag}")
    fell = out["fell_rate"].mean().item()
    ok = ok and fell == 0.0
    print(f"    fell below horizontal: {fell:.2f} | worst |err| = {worst:.3f}")
    return ok, worst, fell


def measure_features(best, ref, inputs, scale):
    out = P.simulate(best.unsqueeze(0), ref, inputs, scale, HIDDEN,
                     episodes=4, init_seed=99, return_features=True)
    print("    signal        p50       p90       p99       max")
    meas = {}
    for f in P.ALL_FEATURES:
        a = out["features"][f].abs().flatten()
        vals = [torch.quantile(a, q).item() for q in (0.5, 0.9, 0.99)] + [a.max().item()]
        meas[f] = dict(p50=vals[0], p90=vals[1], p99=vals[2], max=vals[3])
        print(f"    {f:>9} " + " ".join(f"{v:9.4f}" for v in vals))
    return meas


def main():
    os.makedirs("results_pd", exist_ok=True)
    ref = P.get_reference(device=DEV)
    results = {}

    for i, (name, inputs) in enumerate(CONFIGS.items()):
        scale = P.scale_tensor(inputs, device=DEV)
        n_genes = P.gene_count(len(inputs), HIDDEN)
        path = f"results_pd/baseline_{name}_best.pt"
        print(f"\n{'='*60}\nCONFIG '{name}': inputs={inputs}  genes={n_genes}")
        print("  scales: " + ", ".join(f"{k}={P.DEFAULT_SCALE[k]:.3g}" for k in inputs))

        if os.path.exists(path):
            best = torch.load(path).to(DEV)
            _, fit = select_on_battery(best.unsqueeze(0), inputs, scale)
            newly = False
            print(f"  loaded cached best | battery cost = {fit:.1f}")
        else:
            space = get_space(n_genes, -1.0, 1.0, device=DEV)
            torch.manual_seed(SEED)
            ga = IslandGA(n_islands=N_ISLANDS, pop_size=POP, space=space,
                          evaluate_fn=evaluate_builder(inputs, scale),
                          evaluate_detail_fn=None, num_generations=GENS,
                          migration_interval=MIG, reset_interval=RESET, device=DEV,
                          log_dir=f"results_pd/baseline_{name}_log",
                          fitness_fn="sum|e| + 10 sum|dth|")
            t0 = time.time()
            ga.run()
            final_pop = torch.cat(ga.populations, dim=0)
            best, fit = select_on_battery(final_pop, inputs, scale)
            torch.save(best, path)
            newly = True
            print(f"\n  '{name}' trained in {time.time()-t0:.0f}s | battery cost = {fit:.1f}")

        print("  Tracking (held-out reference):")
        ok, worst, fell = tracking_report(best, ref, inputs, scale)
        meas = measure_features(best, ref, inputs, scale) if newly or i == 0 else None
        results[name] = dict(inputs=list(inputs), genes=n_genes, fitness=fit,
                             tracks_ok=ok, worst_abs_err=worst, fell=fell, measure=meas)

    with open("results_pd/baseline_compare.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n{'='*60}\nSUMMARY")
    print(f"  {'config':>7} {'#in':>4} {'genes':>6} {'battery':>9} {'worst|err|':>11} {'fell':>6} {'tracks':>7}")
    for name, r in results.items():
        print(f"  {name:>7} {len(r['inputs']):>4} {r['genes']:>6} {r['fitness']:>9.1f}"
              f" {r['worst_abs_err']:>11.3f} {r['fell']:>6.2f} {str(r['tracks_ok']):>7}")


if __name__ == "__main__":
    main()
