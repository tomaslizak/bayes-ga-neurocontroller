"""Wall-clock measurements on an idle GPU (work package A10).

For every controller and plant used in the revision, time one fitness evaluation of
  (a) the deterministic GA population   (2700 individuals x 3 episodes),
  (b) the sigma population              (2700 individuals x K = 16 samples x 2 episodes),
and time gradient-VI iterations. A generation of the GA costs one evaluation plus a
negligible amount for the variation operators (checked against complete runs).

Run:   python rev_timing.py        # with nothing else on the GPU
Output: results_rev/timing.json
"""

import json
import time

import numpy as np
import torch

from ga_torch import pendulum as P
from ga_torch import mountaincar_track as T
from ga_torch import bayesian as B
import rev_gradvi as G
from rev_pendulum import Bench, DEV, OUT

N_IND, K, REPS = 2700, 16, 12


def timed(fn, reps=REPS, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def main():
    info = json.load(open(f"{OUT}/mct_baseline.json"))
    cases = [
        ("pendulum GRU-8", Bench()),
        ("pendulum MLP 16-16", Bench(f"{OUT}/baseline_mlp_no_ie.pt", hidden=(16, 16), arch="mlp")),
        ("pendulum GRU-16", Bench(f"{OUT}/baseline_gru16.pt", hidden=16)),
        ("mountain car GRU-8", Bench(f"{OUT}/mct_baseline.pt", inputs=("e", "x", "v"), env=T,
                                     scale_map=info["scale"])),
    ]
    rng = np.random.default_rng(0)
    out = {"gpu": torch.cuda.get_device_name(0), "n_individuals": N_IND, "K": K, "cases": {}}
    print(f"GPU: {out['gpu']}")
    for name, b in cases:
        env, n_w = b.env, b.mu.shape[0]
        pop = torch.rand(N_IND, n_w, device=DEV) * 2 - 1
        sig = 1e-3 + 0.05 * torch.rand(N_IND, n_w, device=DEV)
        genes = torch.cat([b.mu.unsqueeze(0).expand(N_IND, -1), sig], dim=1)

        def det():
            refs = torch.stack([env.get_random_reference(rng=rng, device=DEV) for _ in range(3)])
            return env.simulate(pop, refs, b.inputs, b.scale, b.hidden, 3, arch=b.arch)["cost"]

        def bay():
            refs = torch.stack([env.get_random_reference(rng=rng, device=DEV) for _ in range(2)])
            o = env.simulate_mc(genes, refs, b.inputs, b.scale, b.hidden, K=K, episodes=2,
                                arch=b.arch)
            return o["expected_cost"] + 0.003 * B.kl_divergence(o["mu"], o["sigma"], 1.0)

        t_det, t_bay = timed(det), timed(bay)
        rec = dict(n_weights=n_w, s_per_generation_baseline=t_det, s_per_generation_sigma=t_bay,
                   baseline_1000_generations_s=1000 * t_det, sigma_800_generations_s=800 * t_bay,
                   plant_steps_per_s_sigma=N_IND * K * 2 * env.NSTEPS / t_bay)
        out["cases"][name] = rec
        print(f"{name:<20} {n_w:>5} weights | baseline {t_det * 1e3:6.1f} ms/gen -> "
              f"{1000 * t_det:5.0f} s per 1000 gen | sigma {t_bay * 1e3:6.1f} ms/gen -> "
              f"{800 * t_bay:5.0f} s per 800 gen | {rec['plant_steps_per_s_sigma'] / 1e6:5.0f}M "
              f"plant steps/s", flush=True)

    # gradient VI on pendulum GRU-8: seconds per iteration, 1 seed and 5 seeds in parallel
    b = cases[0][1]
    for n_seeds in (1, 5):
        st = [dict(K=16, lr=0.003, clip=1.0)]
        t0 = time.perf_counter()
        G.optimise(b, st, list(range(n_seeds)), 60, 0.01, eval_every=10 ** 9)   # warm-up + compile
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        G.optimise(b, st, list(range(n_seeds)), 200, 0.01, eval_every=10 ** 9)
        torch.cuda.synchronize()
        per_it = (time.perf_counter() - t1) / 200
        out[f"gradvi_s_per_iteration_{n_seeds}seed"] = per_it
        print(f"gradient VI, {n_seeds} seed(s) in one batch: {per_it * 1e3:6.1f} ms/iteration -> "
              f"{3000 * per_it:5.0f} s per 3000 iterations", flush=True)
    with open(f"{OUT}/timing.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("saved results_rev/timing.json")


if __name__ == "__main__":
    main()
