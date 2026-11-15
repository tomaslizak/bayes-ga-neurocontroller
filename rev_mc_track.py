"""Second benchmark: position tracking on MountainCarContinuous-v0 dynamics (B4).

The pendulum pipeline on a second plant, with nothing changed in the method:
  baseline  island GA (9 x 300, 1000 generations, 3 episodes per evaluation) trains a GRU-8
            controller on the inputs (e, x, v); random set-point sequences per generation,
            battery selection, held-out reference for reporting. Input scaling by the
            1/p99 rule in two passes (rough scales first, then the measured ones).
            The selected controller is also run in the real gymnasium environment.
  sigma     mu fixed, sigma evolved against the negative ELBO with `rev_pendulum.run_one`
            (the same function that runs the pendulum experiments).

Run:   python rev_mc_track.py baseline
       python rev_mc_track.py sigma --lams 0.001,0.003,0.01 --seeds 0
Outputs: results_rev/mct_baseline_rough.*, results_rev/mct_baseline.*,
         results_rev/runs/mct_<tag>.json, results_rev/sigma/mct_<tag>.pt
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from ga_torch import IslandGA
from ga_torch import mountaincar_track as T
from ga_torch.population import get_space
from ga_torch.pendulum import _gru_step
from ga_torch.simulation import unpack_gru_weights
from rev_pendulum import Bench, DEV, OUT, run_one

INPUTS, HIDDEN = ("e", "x", "v"), 8
N_ISLANDS, POP, MIG, GENS_BASE = 9, 300, 40, 1000


def native_deviation(mu, scale, ref, x0):
    """Run the controller in the real gymnasium environment from x0 on the given reference
    and return the largest |x_native - x_port| over the episode (and whether the native
    episode ended early). End-to-end check of plant and task code."""
    import gymnasium as gym
    port = T.simulate(mu.unsqueeze(0), ref, INPUTS, scale, HIDDEN, episodes=1,
                      return_traj=True)          # only used for shapes; re-run below from x0
    env = gym.make("MountainCarContinuous-v0")
    env.reset(seed=0)
    env.unwrapped.state = np.array([x0, 0.0], dtype=np.float32)
    W = unpack_gru_weights(mu.detach().cpu().reshape(1, -1), len(INPUTS), HIDDEN)
    sc = scale.detach().cpu()
    r = ref.cpu().numpy()
    h = torch.zeros(1, HIDDEN)
    x, v = float(x0), 0.0
    xs, ended = [], False
    for k in range(len(r)):
        xs.append(x)
        cur = {"e": r[k] - x, "x": x, "v": v}
        x_t = torch.tensor([[cur[n] for n in INPUTS]], dtype=torch.float32) * sc
        h, a = _gru_step(x_t, h, *W, 1.0)
        obs, _, term, trunc, _ = env.step(np.array([a.item()], dtype=np.float32))
        x, v = float(obs[0]), float(obs[1])
        if term or trunc:
            ended = True
            break
    env.close()
    # the same episode in the batched port, from the same start
    N = 1
    Wg = unpack_gru_weights(mu.reshape(1, -1), len(INPUTS), HIDDEN)
    xp = torch.tensor([x0], device=DEV, dtype=torch.float32); vp = torch.zeros(1, device=DEV)
    hp = torch.zeros(N, HIDDEN, device=DEV)
    xs_port = []
    for k in range(len(xs)):
        xs_port.append(xp.item())
        cur = {"e": ref[k] - xp, "x": xp, "v": vp}
        x_t = torch.stack([cur[n] for n in INPUTS], dim=1) * scale
        hp, a = _gru_step(x_t, hp, *Wg, 1.0)
        xp, vp, _ = T.plant_step(xp, vp, a)
    return float(np.abs(np.array(xs) - np.array(xs_port)).max()), ended


def train_baseline(which, seed=0, gens=GENS_BASE):
    name = "mct_baseline_rough" if which == "rough" else "mct_baseline"
    path = f"{OUT}/{name}.pt"
    if os.path.exists(path):
        print(f"{path} exists, skipping training"); return
    os.makedirs(OUT, exist_ok=True)
    if which == "rough":
        scale_map = dict(T.ROUGH_SCALE)
    else:
        m = json.load(open(f"{OUT}/mct_baseline_rough.json"))["p99"]
        scale_map = {k: 1.0 / m[k] for k in INPUTS}
    scale = T.scale_tensor(INPUTS, scale_map, device=DEV)
    battery = T.validation_battery(device=DEV)
    n_genes = T.gene_count(len(INPUTS), HIDDEN)
    rng = np.random.default_rng(12345 + seed)

    def evaluate(genes):
        refs = torch.stack([T.get_random_reference(rng=rng, device=DEV) for _ in range(3)])
        return T.simulate(genes, refs, INPUTS, scale, HIDDEN, 3)["cost"]

    torch.manual_seed(seed)
    ga = IslandGA(n_islands=N_ISLANDS, pop_size=POP,
                  space=get_space(n_genes, -1.0, 1.0, device=DEV),
                  evaluate_fn=evaluate, evaluate_detail_fn=None,
                  num_generations=gens, migration_interval=MIG, reset_interval=201,
                  device=DEV, log_dir=f"{OUT}/{name}_log", fitness_fn="baseline")
    t0 = time.time()
    ga.run()
    t_ga = time.time() - t0
    final = torch.cat(ga.populations, dim=0)
    out = [T.simulate(final[i:i + 512], battery, INPUTS, scale, HIDDEN,
                      episodes=battery.shape[0], init_seed=555) for i in range(0, len(final), 512)]
    costs = torch.cat([o["cost"] for o in out]); fell = torch.cat([o["fell_rate"] for o in out])
    best = final[costs.argmin()].clone()
    torch.save(best.cpu(), path)

    # held-out reference: tracking quality, signal percentiles, cost split
    ref = T.get_reference(device=DEV)
    tr = T.simulate(best.unsqueeze(0), ref, INPUTS, scale, HIDDEN, episodes=4, init_seed=99,
                    return_traj=True, return_features=True)
    x = tr["x"].cpu().numpy(); u = tr["u"].cpu().numpy(); r = ref.cpu().numpy()
    k = np.arange(len(r))
    settled = np.zeros(len(r), bool)
    for i in range(4):                                  # last 40 steps of every level
        s = T.SETTLE + T.SEGMENT * i
        settled |= (k >= s + T.SEGMENT - 40) & (k < s + T.SEGMENT)
    track = float(np.abs(r - x).sum(1).mean())
    smooth = float(T.ALFA * np.abs(np.diff(x, axis=1)).sum(1).mean())
    lvl1 = (k >= T.SETTLE + 60) & (k < T.SETTLE + 100)   # steady part of the +0.125 level
    dev, ended = native_deviation(best, scale, ref, x0=-0.5)
    info = dict(which=which, seed=seed, gens=gens, n_weights=n_genes, scale=scale_map,
                battery_cost=costs.min().item(), battery_fell=fell[costs.argmin()].item(),
                heldout_cost=tr["cost"][0].item(), heldout_tracking_part=track,
                heldout_smoothness_part=smooth,
                heldout_steady_state_err=float(np.abs(x[:, settled] - r[settled]).max()),
                peak_abs_u=float(np.abs(u).max()), steady_u_at_0125=float(u[:, lvl1].mean()),
                p99={f: float(torch.quantile(tr["features"][f].abs().flatten(), 0.99))
                     for f in T.ALL_FEATURES},
                native_max_deviation=dev, native_ended_early=ended, t_ga_s=t_ga)
    with open(f"{OUT}/{name}.json", "w") as fh:
        json.dump(info, fh, indent=2)
    print(f"### {name}: battery cost {info['battery_cost']:.2f} (fell {info['battery_fell']:.3f}) | "
          f"held-out: cost {info['heldout_cost']:.2f} = tracking {track:.2f} + smoothness "
          f"{smooth:.2f}, steady-state error {info['heldout_steady_state_err']:.4f}, peak |u| "
          f"{info['peak_abs_u']:.2f}, steady u at +0.125: {info['steady_u_at_0125']:.3f} "
          f"(theory {(T.PHYS['gravity'] / T.PHYS['power']) * np.sin(3 * 0.125):.3f}) | native "
          f"gymnasium: max |x deviation| {dev:.2e}, ended early: {ended} | p99 "
          f"{ {f: round(val, 4) for f, val in info['p99'].items()} } ({t_ga:.0f}s)", flush=True)


def bench():
    info = json.load(open(f"{OUT}/mct_baseline.json"))
    return Bench(f"{OUT}/mct_baseline.pt", hidden=HIDDEN, arch="gru", inputs=INPUTS, env=T,
                 scale_map=info["scale"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["baseline", "sigma"])
    ap.add_argument("--lams", default="0.001,0.003,0.01,0.03")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--plan", default="self_adaptive", choices=["self_adaptive", "default"])
    args = ap.parse_args()
    if args.stage == "baseline":
        train_baseline("rough")
        train_baseline("measured")
        return
    b = bench()
    c = b.crit
    print(f"floor {b.floor:.3f} | criticality p05 {np.quantile(c, .05):.4f} p50 "
          f"{np.median(c):.4f} p95 {np.quantile(c, .95):.2f} max {c.max():.2f} | share of "
          f"weights with c < 1e-3: {(c < 1e-3).mean():.2f}", flush=True)
    for seed in [int(s) for s in args.seeds.split(",")]:
        for lam in [float(x) for x in args.lams.split(",")]:
            tag = f"mct_lam{lam:g}_s{seed}" + ("_stdops" if args.plan == "default" else "")
            run_one(b, tag, seed, lam=lam, plan=args.plan)


if __name__ == "__main__":
    main()
