"""CartPole pipeline, step 1: train the deterministic GRU tracking baseline on native
gymnasium CartPole-v1, verify it tracks the cart-position set-points, and measure the
real range of every candidate input for the 1/p99 scaling rule.

Mirrors train_baseline.py (ODE plant). Configs are compared under the same seed/budget.

Run:  python train_cartpole.py
Outputs: results_cp/baseline_<name>_best.pt, results_cp/baseline_compare.json
"""

import json
import os
import time

import numpy as np
import torch

from ga_torch import IslandGA
from ga_torch import cartpole as CP
from ga_torch.population import get_space
from ga_torch.simulation import setup_precision

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
setup_precision("high")

# name -> input tuple. First config also produces the p99 measurement.
CONFIGS = {
    "all6":  ("e", "ie", "x", "x_dot", "theta", "theta_dot"),
    "no_x":  ("e", "ie", "x_dot", "theta", "theta_dot"),   # is absolute position redundant given e?
    "no_ie": ("e", "x", "x_dot", "theta", "theta_dot"),    # is the error integral needed?
}
HIDDEN = 8
SEED = 0
EPISODES_TRAIN = 3          # episodes (each with its own random reference) per evaluation
N_ISLANDS, POP, GENS, MIG, RESET = 9, 300, 1000, 40, 201

# reference segments (start_s, end_s, setpoint) for the steady-state check
SEGMENTS = [(2, 9, 1.5), (9, 16, -1.0), (16, 23, 0.5), (23, 30, 0.0)]


def evaluate_builder(ref, inputs, scale):
    # Train on FRESH random set-point sequences every evaluation (one per episode slot,
    # identical for all individuals within the call, so within-generation ranking is
    # fair). This prevents the GRU from memorising a reference timetable instead of
    # learning feedback control. NOTE: fitness values are NOT comparable across
    # generations (references differ in difficulty) -- final selection must use the
    # fixed validation battery, not the GA's "best ever" bookkeeping.
    rng = np.random.default_rng(12345)
    def evaluate(genes):
        train_refs = torch.stack([CP.get_random_reference(rng=rng, device=genes.device)
                                  for _ in range(EPISODES_TRAIN)])
        return CP.simulate(genes, train_refs, inputs, scale, HIDDEN, EPISODES_TRAIN)["cost"]
    return evaluate


def select_on_battery(final_pop, inputs, scale, chunk=512):
    """Re-rank a population on the fixed validation battery; return (best, val_cost)."""
    battery = CP.validation_battery(device=final_pop.device)
    E = battery.shape[0]
    costs = []
    for i in range(0, final_pop.shape[0], chunk):
        costs.append(CP.simulate(final_pop[i:i + chunk], battery, inputs, scale,
                                 HIDDEN, episodes=E, init_seed=555)["cost"])
    costs = torch.cat(costs)
    idx = costs.argmin()
    return final_pop[idx].clone(), costs[idx].item()


def tracking_report(best, ref, inputs, scale):
    """Mean cart position over the last 2 s of each segment vs its set-point.

    Averaged over the SURVIVING evaluation episodes only: after an episode dies the
    environment autoresets and its trajectory belongs to a fresh episode, so including
    it would corrupt the numbers. Deaths are reported separately.
    """
    out = CP.simulate(best.unsqueeze(0), ref, inputs, scale, HIDDEN,
                      episodes=4, init_seed=99, return_traj=True)
    died = out["blown_rate"].mean().item()
    survived = out["alive_mask"].flatten()
    print(f"    episodes survived: {int(survived.sum())}/4")
    if survived.sum() == 0:
        print("    (no surviving episode -- tracking not measurable)")
        return False, float("inf"), died
    x = out["x"].reshape(4, -1)[survived].mean(dim=0)
    print("    segment(s)  setpoint  achieved      err")
    ok, worst = True, 0.0
    for (a, b, sp) in SEGMENTS:
        i0, i1 = int((b - 2) / CP.TAU), int(b / CP.TAU)
        ach = x[i0:i1].mean().item()
        err = ach - sp
        worst = max(worst, abs(err))
        flag = "" if abs(err) < 0.15 else "  <-- off"
        ok = ok and not flag
        print(f"    [{a:>3},{b:>3}) {sp:8.2f} {ach:9.3f} {err:8.3f}{flag}")
    ok = ok and died == 0.0
    print(f"    episodes died early: {died:.2f} | worst |err| = {worst:.3f}")
    return ok, worst, died


def measure_features(best, ref, inputs, scale):
    out = CP.simulate(best.unsqueeze(0), ref, inputs, scale, HIDDEN,
                      episodes=4, init_seed=99, return_features=True)
    feats = out["features"]
    print("    signal        p50       p90       p99       max")
    meas = {}
    for f in CP.ALL_FEATURES:
        a = feats[f].abs().flatten()
        vals = [torch.quantile(a, q).item() for q in (0.5, 0.9, 0.99)] + [a.max().item()]
        meas[f] = dict(p50=vals[0], p90=vals[1], p99=vals[2], max=vals[3],
                       suggest_scale=(1.0 / vals[2]) if vals[2] > 0 else None)
        print(f"    {f:>9} " + " ".join(f"{v:9.3f}" for v in vals))
    return meas


def main():
    os.makedirs("results_cp", exist_ok=True)
    ref = CP.get_reference(device=DEV)
    results = {}

    for i, (name, inputs) in enumerate(CONFIGS.items()):
        scale = CP.scale_tensor(inputs, device=DEV)
        n_genes = CP.gene_count(len(inputs), HIDDEN)
        path = f"results_cp/baseline_{name}_best.pt"
        print(f"\n{'='*60}\nCONFIG '{name}': inputs={inputs}  genes={n_genes}")
        print("  scales: " + ", ".join(f"{k}={CP.DEFAULT_SCALE[k]:.3g}" for k in inputs))

        if os.path.exists(path):
            best = torch.load(path).to(DEV)
            fit = CP.simulate(best.unsqueeze(0), ref, inputs, scale, HIDDEN,
                              episodes=8, init_seed=99)["cost"][0].item()
            newly = False
            print(f"  loaded cached best | cost(8 fixed episodes) = {fit:.1f}")
        else:
            space = get_space(n_genes, -1.0, 1.0, device=DEV)
            torch.manual_seed(SEED)
            ga = IslandGA(n_islands=N_ISLANDS, pop_size=POP, space=space,
                          evaluate_fn=evaluate_builder(ref, inputs, scale),
                          evaluate_detail_fn=None, num_generations=GENS,
                          migration_interval=MIG, reset_interval=RESET, device=DEV,
                          log_dir=f"results_cp/baseline_{name}_log",
                          fitness_fn="sum|e|+10|dx| (alive) + 6/step (dead)")
            t0 = time.time()
            ga.run()
            # "best ever" is meaningless under per-generation random references --
            # select from the final population on the fixed validation battery.
            final_pop = torch.cat(ga.populations, dim=0)
            best, fit = select_on_battery(final_pop, inputs, scale)
            torch.save(best, path)
            newly = True
            print(f"\n  '{name}' trained in {time.time()-t0:.0f}s | "
                  f"validation-battery cost = {fit:.1f}")

        print("  Tracking:")
        ok, worst, blown = tracking_report(best, ref, inputs, scale)
        meas = measure_features(best, ref, inputs, scale) if newly or i == 0 else None
        results[name] = dict(inputs=list(inputs), genes=n_genes, fitness=fit,
                             tracks_ok=ok, worst_abs_err=worst, blown=blown, measure=meas)

    with open("results_cp/baseline_compare.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n{'='*60}\nSUMMARY")
    print(f"  {'config':>8} {'#in':>4} {'genes':>6} {'fitness':>10} {'worst|err|':>11} {'died':>6} {'tracks':>7}")
    for name, r in results.items():
        print(f"  {name:>8} {len(r['inputs']):>4} {r['genes']:>6} {r['fitness']:>10.1f}"
              f" {r['worst_abs_err']:>11.3f} {r['blown']:>6.2f} {str(r['tracks_ok']):>7}")


if __name__ == "__main__":
    main()
