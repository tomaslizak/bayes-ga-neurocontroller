"""Other controller architectures on Pendulum-v1 (work packages B2 and B3).

Same plant, task, input scaling, GA budget and protocol as the paper's GRU-8 (272 weights);
only the controller changes:
  --hidden 16 [32]   GRU with a larger hidden state (928 / 3392 weights)
  --mlp              feedforward MLP in -> 16 -> 16 -> 1, tanh, with biases (353 weights
                     for three inputs). The input set is chosen by the rule of Table 1:
                     with and without the error integral, lower battery cost wins.
For each controller:
  1. deterministic baseline: island GA, 1000 generations, battery-selected;
  2. criticality of every weight;
  3. sigma evolution at the paper's settings (lambda 0.003), several seeds, scored with the
     common scorer of rev_pendulum.py.

Run:   python rev_arch.py --hidden 16 --seeds 3
       python rev_arch.py --mlp --seeds 0      # baselines only
       python rev_arch.py --mlp --seeds 3
Outputs: results_rev/baseline_<name>.pt/.json, results_rev/runs/<name>_s<seed>.json,
         results_rev/sigma/<name>_s<seed>.pt
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from ga_torch import IslandGA
from ga_torch import pendulum as P
from ga_torch.population import get_space
from rev_pendulum import Bench, DEV, INPUTS, OUT, run_one

N_ISLANDS, POP, MIG, GENS_BASE = 9, 300, 40, 1000
MLP_LAYERS = (16, 16)
INPUTS_IE = ("e", "ie", "theta", "theta_dot")


def train_baseline(name, inputs, hidden, arch, seed=0, gens=GENS_BASE):
    """Train (or load) a deterministic controller; identical recipe to the paper."""
    path = f"{OUT}/baseline_{name}.pt"
    if os.path.exists(path):
        return path
    os.makedirs(OUT, exist_ok=True)
    scale = P.scale_tensor(inputs, device=DEV)
    battery = P.validation_battery(device=DEV)
    n_genes = P.gene_count(len(inputs), hidden, arch)
    rng = np.random.default_rng(12345 + seed)

    def evaluate(genes):
        refs = torch.stack([P.get_random_reference(rng=rng, device=DEV) for _ in range(3)])
        return P.simulate(genes, refs, inputs, scale, hidden, 3, arch=arch)["cost"]

    torch.manual_seed(seed)
    ga = IslandGA(n_islands=N_ISLANDS, pop_size=POP,
                  space=get_space(n_genes, -1.0, 1.0, device=DEV),
                  evaluate_fn=evaluate, evaluate_detail_fn=None,
                  num_generations=gens, migration_interval=MIG, reset_interval=201,
                  device=DEV, log_dir=f"{OUT}/baseline_{name}_log", fitness_fn="baseline")
    t0 = time.time()
    ga.run()
    t_ga = time.time() - t0
    final = torch.cat(ga.populations, dim=0)
    out = [P.simulate(final[i:i + 512], battery, inputs, scale, hidden,
                      episodes=battery.shape[0], init_seed=555, arch=arch)
           for i in range(0, len(final), 512)]
    costs = torch.cat([o["cost"] for o in out]); fell = torch.cat([o["fell_rate"] for o in out])
    best = costs.argmin()
    torch.save(final[best].clone().cpu(), path)

    # tracking quality on the held-out reference (never used for training or selection)
    ref = P.get_reference(device=DEV)
    tr = P.simulate(final[best].unsqueeze(0), ref, inputs, scale, hidden, episodes=4,
                    init_seed=99, return_traj=True, arch=arch)
    th = tr["theta"].cpu().numpy(); r = ref.cpu().numpy()
    t = np.arange(len(r)) * P.DT
    settled = np.zeros(len(r), bool)
    for start in (2, 7, 12, 17):                       # last 2 s of every 5 s level
        settled |= (t >= start + 3) & (t < start + 5)
    info = dict(name=name, arch=arch, hidden=hidden, inputs=list(inputs), n_weights=n_genes,
                seed=seed, gens=gens, battery_cost=costs[best].item(),
                battery_fell=fell[best].item(), heldout_cost=tr["cost"][0].item(),
                heldout_steady_state_err=float(np.abs(th[:, settled] - r[settled]).max()),
                t_ga_s=t_ga)
    with open(f"{OUT}/baseline_{name}.json", "w") as fh:
        json.dump(info, fh, indent=2)
    print(f"### baseline {name}: {n_genes} weights, battery cost {info['battery_cost']:.2f}, "
          f"fell {info['battery_fell']:.3f}, held-out steady-state error "
          f"{info['heldout_steady_state_err']:.4f} rad ({t_ga:.0f}s)", flush=True)
    return path


def sigma_runs(bench, name, n_seeds, gens_sigma, out_tag=""):
    c = bench.crit
    print(f"{name}: floor {bench.floor:.2f} | criticality p50 {np.median(c):.4f} "
          f"max {c.max():.2f} | share of weights with c < 1e-3: {(c < 1e-3).mean():.2f}",
          flush=True)
    for s in range(n_seeds):
        run_one(bench, f"{name}{out_tag}_s{s}", s, gens=gens_sigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, nargs="*", default=[])
    ap.add_argument("--mlp", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gens-base", type=int, default=GENS_BASE)
    ap.add_argument("--gens-sigma", type=int, default=800)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    for h in args.hidden:
        path = train_baseline(f"gru{h}", INPUTS, h, "gru", gens=args.gens_base)
        sigma_runs(Bench(path, hidden=h), f"gru{h}", args.seeds, args.gens_sigma, args.out_tag)

    if args.mlp:
        cands = {"mlp_no_ie": INPUTS, "mlp_ie": INPUTS_IE}
        costs = {}
        for name, inputs in cands.items():
            train_baseline(name, inputs, MLP_LAYERS, "mlp", gens=args.gens_base)
            costs[name] = json.load(open(f"{OUT}/baseline_{name}.json"))["battery_cost"]
        chosen = min(costs, key=costs.get)                 # Table 1 rule
        print(f"MLP input choice: {costs} -> {chosen}", flush=True)
        with open(f"{OUT}/baseline_mlp_choice.json", "w") as fh:
            json.dump(dict(battery_cost=costs, chosen=chosen, inputs=list(cands[chosen])), fh,
                      indent=2)
        if args.seeds > 0:
            bench = Bench(f"{OUT}/baseline_{chosen}.pt", hidden=MLP_LAYERS, arch="mlp",
                          inputs=cands[chosen])
            sigma_runs(bench, "mlp", args.seeds, args.gens_sigma, args.out_tag)


if __name__ == "__main__":
    main()
