"""Gradient-based variational inference baseline on Pendulum-v1 (work package B1).

Bayes by Backprop (Blundell et al. 2015) applied to the same problem the GA solves: mu is
fixed at the trained controller, the per-weight sigma is optimised against the same
negative ELBO,  E_q[J(w)] + lambda * KL(q || N(0, sigma_p^2 I)),  with the
reparameterisation w = mu + sigma * eps and gradients backpropagated through the closed-loop
rollout (the torch pendulum is differentiable). Same sigma bounds and the same random
references per step as one GA fitness evaluation. Several seeds are optimised in parallel
as independent rows of one batch.

The rollout is `pendulum.simulate` itself with gradients enabled, so plant, controller and
cost are identical to the GA runs by construction.

To give the baseline a fair chance, the learning rate, the number of Monte-Carlo samples
per step and the gradient-clipping threshold are swept, and the iterate is selected on the
frozen validation battery (the GA result is battery-selected too). Both the last and the
battery-selected iterate are reported, together with gradient-norm statistics.

Run:   python rev_gradvi.py --sweep
       python rev_gradvi.py --K 16 --lr 0.03 --clip 1   # one setting
Output: results_rev/gradvi_sweep.json (or gradvi_<setting>.json), results_rev/sigma/gradvi_*.pt
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from ga_torch import pendulum as P
from ga_torch import bayesian as B
from rev_pendulum import Bench, DEV, HIDDEN, INPUTS, spearman

LAM, SIGMA_PRIOR = 0.003, 1.0
SIGMA_MIN, SIGMA_MAX = 1e-3, 0.3
EPISODES = 2
K_EVAL = 256                                     # battery samples for iterate selection

simulate_grad = P.simulate.__wrapped__          # pendulum.simulate without torch.no_grad
PREFIX = ""                                     # set by --name for other baselines


def main_sigma_path(seed):
    legacy = f"results_pd/sigma_lam0.003_s{seed}_best.pt"
    return legacy if seed < 3 else f"results_pd/sigma_b0_s{seed}_best.pt"


def check_same_rollout(bench):
    """The gradient-enabled rollout must return exactly the costs of the GA's rollout."""
    g = torch.Generator(device=DEV).manual_seed(1)
    W = bench.mu.unsqueeze(0) + 0.01 * torch.randn(64, bench.mu.shape[0], device=DEV,
                                                   generator=g)
    a = P.simulate(W, bench.battery, INPUTS, bench.scale, bench.hidden, bench.E, init_seed=555)["cost"]
    b = simulate_grad(W, bench.battery, INPUTS, bench.scale, bench.hidden, bench.E,
                      init_seed=555)["cost"]
    return float((a - b).abs().max())


def battery_neg_elbo(bench, sigma):
    """Negative ELBO of each row of sigma on the frozen battery, common random numbers."""
    gen = torch.Generator(device=DEV).manual_seed(4321)
    S = sigma.shape[0]
    genes = torch.cat([bench.mu.unsqueeze(0).expand(S, -1), sigma], dim=1)
    o = P.simulate_mc(genes, bench.battery, INPUTS, bench.scale, bench.hidden, K=K_EVAL,
                      episodes=bench.E, init_seed=555, generator=gen)
    kl = B.kl_divergence(o["mu"], o["sigma"], SIGMA_PRIOR)
    return o["expected_cost"] + LAM * kl, o["expected_cost"], o["fell_rate"], kl


def optimise(bench, settings, seeds, iters, init_sigma, eval_every=100):
    """Train every setting (dict with K, lr, clip) for every seed in ONE batched pass.

    Each (setting, seed) pair is an independent run with its own parameters, optimiser and
    posterior samples; only the forward/backward pass through the rollout is shared, which
    costs almost no extra wall time on the GPU. Random references are shared, as they are
    shared by all individuals of a GA generation.
    """
    S, n_w = len(seeds), bench.mu.shape[0]
    lo, hi = math.log(SIGMA_MIN), math.log(SIGMA_MAX)
    gen = torch.Generator(device=DEV).manual_seed(1000 + seeds[0])
    rng = np.random.default_rng(4242 + seeds[0])
    mu = bench.mu.reshape(1, 1, -1)
    mu_rows = bench.mu.unsqueeze(0).expand(S, -1)
    runs = []
    for st in settings:
        logsig = torch.full((S, n_w), math.log(init_sigma), device=DEV, requires_grad=True)
        opt = torch.optim.Adam([logsig], lr=st["lr"])
        runs.append(dict(st=st, logsig=logsig, opt=opt,
                         sched=torch.optim.lr_scheduler.CosineAnnealingLR(
                             opt, T_max=iters, eta_min=st["lr"] / 10),
                         best_f=torch.full((S,), float("inf"), device=DEV),
                         best_sig=torch.full((S, n_w), init_sigma, device=DEV),
                         best_it=torch.zeros(S, dtype=torch.long, device=DEV),
                         hist=[], norms=[], fells=[], nonfinite=0))
    t0 = time.time()
    for it in range(1, iters + 1):
        refs = torch.stack([P.get_random_reference(rng=rng, device=DEV)
                            for _ in range(EPISODES)])
        Ws, sigmas = [], []
        for r in runs:
            k = r["st"]["K"]
            sigma = r["logsig"].exp(); sigmas.append(sigma)
            eps = torch.randn(S, k, n_w, device=DEV, generator=gen)
            Ws.append((mu + sigma.unsqueeze(1) * eps).reshape(S * k, n_w))
        out = simulate_grad(torch.cat(Ws), refs, INPUTS, bench.scale, bench.hidden, EPISODES)
        costs = out["cost"].clamp(max=B.COST_CLIP).split([w.shape[0] for w in Ws])
        fell = out["fell_rate"].split([w.shape[0] for w in Ws])
        loss = 0.0
        for r, sigma, c in zip(runs, sigmas, costs):
            data = c.reshape(S, r["st"]["K"]).mean(dim=1)
            loss = loss + (data + LAM * B.kl_divergence(mu_rows, sigma, SIGMA_PRIOR)).sum()
        for r in runs:
            r["opt"].zero_grad(set_to_none=True)
        loss.backward()
        for r, f in zip(runs, fell):
            g, clip = r["logsig"].grad, r["st"]["clip"]
            r["nonfinite"] += int((~torch.isfinite(g)).any().item())
            g.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            norm = g.norm(dim=1, keepdim=True)
            r["norms"].append(norm.squeeze(1).cpu()); r["fells"].append(f.mean().item())
            g.mul_((clip / norm.clamp(min=1e-12)).clamp(max=1.0))   # per-run norm clipping
            r["opt"].step(); r["sched"].step()
            with torch.no_grad():
                r["logsig"].clamp_(lo, hi)                # same feasible set as the GA
        if it % eval_every == 0 or it == iters:
            for r in runs:
                with torch.no_grad():
                    sig = r["logsig"].exp()
                    f, ec, fl, klb = battery_neg_elbo(bench, sig)
                    better = f < r["best_f"]
                    r["best_f"] = torch.where(better, f, r["best_f"])
                    r["best_sig"][better] = sig[better]
                    r["best_it"][better] = it
                nn = torch.stack(r["norms"][-eval_every:])
                clip = r["st"]["clip"]
                r["hist"].append(dict(
                    it=it, t=time.time() - t0, sigma_mean=sig.mean().item(),
                    battery_neg_elbo=f.mean().item(), battery_E_cost=ec.mean().item(),
                    battery_fell=fl.mean().item(), kl=klb.mean().item(),
                    grad_norm_median=nn.median().item(), grad_norm_max=nn.max().item(),
                    frac_clipped=float((nn > clip).float().mean()),
                    train_fell=float(np.mean(r["fells"][-eval_every:]))))
            if it % (5 * eval_every) == 0 or it == iters:
                print(f"  it {it} ({time.time() - t0:.0f}s)")
                for r in runs:
                    h, st = r["hist"][-1], r["st"]
                    print(f"    K{st['K']:<4} lr{st['lr']:<6g} clip{st['clip']:<5g} battery "
                          f"-ELBO {h['battery_neg_elbo']:7.2f} fell {h['battery_fell']:.4f} "
                          f"mean sigma {h['sigma_mean']:.4f} | grad norm med "
                          f"{h['grad_norm_median']:.3g} max {h['grad_norm_max']:.3g} clipped "
                          f"{100 * h['frac_clipped']:.0f}%", flush=True)
    wall = time.time() - t0
    return [dict(st=r["st"], last=r["logsig"].detach().exp(), best=r["best_sig"],
                 best_it=r["best_it"].tolist(), hist=r["hist"], wall=wall,
                 nonfinite_iters=r["nonfinite"]) for r in runs]


def describe(bench, sig, ga_med):
    r = bench.score(sig, SIGMA_PRIOR)
    v = sig.cpu().numpy()
    r.update(neg_elbo=r["E_cost"] + LAM * r["kl"],
             rho_with_ga_median=(spearman(v, ga_med) if ga_med is not None
                                 else float("nan")),
             frac_at_cap=float((v >= 0.98 * SIGMA_MAX).mean()),
             frac_at_floor=float((v <= 1.5 * SIGMA_MIN).mean()))
    return r


def summarise(bench, ga_med, res, iters, n_seeds, init):
    st = res["st"]
    tag = f"gradvi{PREFIX}_K{st['K']}_lr{st['lr']:g}_clip{st['clip']:g}"
    print(f"\n===== {tag} =====", flush=True)
    os.makedirs("results_rev/sigma", exist_ok=True)
    rows = []
    for i in range(n_seeds):
        row = dict(seed=i, best_it=res["best_it"][i])
        for which in ("last", "best"):
            sig = res[which][i]
            torch.save(sig.cpu(), f"results_rev/sigma/{tag}_s{i}_{which}.pt")
            row[which] = describe(bench, sig, ga_med)
        rows.append(row)
        b, l = row["best"], row["last"]
        print(f"  seed {i}: selected it {row['best_it']:>4}: -ELBO {b['neg_elbo']:.2f} rho "
              f"{b['rho']:+.3f} fell {b['fell']:.4f} E[J] {b['E_cost']:.2f} mean sigma "
              f"{b['sigma_mean']:.3f} rank-vs-GA {b['rho_with_ga_median']:+.3f} || last: -ELBO "
              f"{l['neg_elbo']:.2f} rho {l['rho']:+.3f} fell {l['fell']:.4f}", flush=True)
    return dict(tag=tag, K=st["K"], lr=st["lr"], clip=st["clip"], iters=iters, init=init,
                runs=rows, history=res["hist"], wall_s_shared_pass=res["wall"],
                nonfinite_iters=res["nonfinite_iters"],
                rollouts_per_seed=iters * st["K"] * EPISODES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--init", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=100.0)
    ap.add_argument("--mu", default="", help="other baseline (.pt); default: the paper's GRU-8")
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--name", default="", help="tag for results of a non-default baseline")
    args = ap.parse_args()
    global PREFIX
    PREFIX = f"_{args.name}" if args.name else ""

    bench = Bench(args.mu, hidden=args.hidden) if args.mu else Bench()
    print(f"floor {bench.floor:.2f}, {bench.mu.shape[0]} weights | rollout check, "
          f"max |cost difference| = {check_same_rollout(bench):.3g}")
    ga_med, ga_scores, pair = None, [], [float("nan")]
    if not args.mu:                 # comparison with the GA runs behind the submitted paper
        ga = np.stack([torch.load(main_sigma_path(s)).cpu().numpy() for s in range(50)])
        ga_med = np.median(ga, axis=0)
        for s in range(5):
            r = bench.score(torch.as_tensor(ga[s]).to(DEV))
            r.update(seed=s, neg_elbo=r["E_cost"] + LAM * r["kl"])
            ga_scores.append(r)
            print(f"GA seed {s}: -ELBO {r['neg_elbo']:.2f} rho {r['rho']:+.3f} fell "
                  f"{r['fell']:.4f} E[J] {r['E_cost']:.2f} mean sigma {r['sigma_mean']:.3f}")
        pair = [spearman(ga[i], ga[j]) for i in range(5) for j in range(i + 1, 5)]

    settings = ([dict(K=k, lr=lr, clip=c) for k in (16, 128) for lr in (0.03, 0.01, 0.003)
                 for c in (1.0, 100.0)]
                if args.sweep else [dict(K=args.K, lr=args.lr, clip=args.clip)])
    trained = optimise(bench, settings, list(range(args.seeds)), args.iters, args.init)
    results = [summarise(bench, ga_med, r, args.iters, args.seeds, args.init) for r in trained]

    print(f"\n{'config':<30}{'-ELBO sel.':>12}{'rho sel.':>10}{'fell sel.':>11}"
          f"{'-ELBO last':>12}{'fell last':>11}{'wall s':>8}")
    for c in results:
        f = lambda w, key: np.mean([r[w][key] for r in c["runs"]])
        print(f"{c['tag']:<30}{f('best', 'neg_elbo'):>12.2f}{f('best', 'rho'):>+10.3f}"
              f"{f('best', 'fell'):>11.4f}{f('last', 'neg_elbo'):>12.2f}"
              f"{f('last', 'fell'):>11.4f}{c['wall_s_shared_pass']:>8.0f}")
    if ga_scores:
        print(f"{'GA (seeds 0-4)':<30}{np.mean([r['neg_elbo'] for r in ga_scores]):>12.2f}"
              f"{np.mean([r['rho'] for r in ga_scores]):>+10.3f}"
              f"{np.mean([r['fell'] for r in ga_scores]):>11.4f}")
    name = f"gradvi{PREFIX}_sweep" if args.sweep else results[0]["tag"]
    with open(f"results_rev/{name}.json", "w") as fh:
        json.dump(dict(lam=LAM, episodes=EPISODES, K_eval=K_EVAL, floor=bench.floor,
                       configs=results, ga_first5=ga_scores,
                       ga_pair_rank_agreement=float(np.mean(pair))), fh, indent=2)
    print(f"saved results_rev/{name}.json")


if __name__ == "__main__":
    main()
