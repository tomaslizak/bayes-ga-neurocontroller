"""Numbers of Tables 2-5 and of the paragraphs on averaging runs, recomputed from the
stored results. The GPU is needed only to score the averaged posteriors (a few seconds).

Sources: results_rev/runs/*.json (one record per run), results_rev/analysis.json (the 50
runs of the first submission re-scored with 2048 samples), results_rev/gradvi_sweep.json,
and the stored sigma vectors. "Averaged" means the geometric mean of the runs' sigma
vectors, scored like a single run.

Run:   python rev_tables.py
"""

import json

import numpy as np
import torch

import rev_mc_track as MC
from rev_pendulum import Bench, DEV, OUT, spearman

LAM = 0.003


def runs(prefix, seeds):
    return [json.load(open(f"{OUT}/runs/{prefix}_s{s}.json")) for s in seeds]


def ms(recs, key, scale=1.0):
    v = np.array([r[key] for r in recs], float) * scale
    return v.mean(), v.std()


def neg_elbo(r):
    return r["E_cost"] + LAM * r["kl"]


def paper_sigma(s):
    """The 50 runs of the first submission (Section 5.2)."""
    return (f"results_pd/sigma_lam0.003_s{s}_best.pt" if s < 3
            else f"results_pd/sigma_b0_s{s}_best.pt")


def load(paths):
    return np.stack([torch.load(p).cpu().numpy() for p in paths])


def averaged(bench, sig):
    s = torch.as_tensor(np.exp(np.log(sig).mean(0)), dtype=torch.float32).to(DEV)
    r = bench.score(s)
    r["neg_elbo"] = neg_elbo(r)
    return r


def agreement(sig):
    n = len(sig)
    return np.mean([spearman(sig[i], sig[j]) for i in range(n) for j in range(i + 1, n)])


def row(label, recs):
    rho, rs = ms(recs, "rho"); fell, _ = ms(recs, "fell", 100)
    ec, es = ms(recs, "E_cost"); sg, _ = ms(recs, "sigma_mean")
    t = np.median([r["t_ga_s"] for r in recs])
    print(f"  {label:<16} n={len(recs)}  rho {rho:+.2f}+-{rs:.2f}  mean sigma {sg:.3f}  "
          f"fall {fell:.2f}%  E[J] {ec:.1f}+-{es:.1f}  time {t:.0f} s")


def main():
    print("Table 2 (pendulum GRU-8, lambda 0.003)")
    row("standard", runs("stdops", range(5)))
    row("self-adaptive", runs("lam0.003", range(5)))

    print("Table 3 (one setting changed at a time)")
    row("default", runs("lam0.003", range(5)))
    for g in ("sp0.1", "sp0.3", "sp3", "sp10", "smax0.1", "smax1", "K2", "K4", "K8", "K32",
              "pop1x100", "pop3x100", "pop9x100", "gens200", "gens400"):
        row(g, runs(g, range(3)))

    print("Table 4 (single runs and averaged runs)")
    a = json.load(open(f"{OUT}/analysis.json"))["rescore"]
    main50 = a["main"]
    b = Bench()
    sig50 = load([paper_sigma(s) for s in range(50)])
    av5 = averaged(b, sig50[:5])
    rho, rs = ms(main50, "rho")
    print(f"  pendulum GRU-8   floor {b.floor:.1f}  rho {rho:+.2f}+-{rs:.2f}  fall "
          f"{ms(main50, 'fell', 100)[0]:.2f}%  E[J] {ms(main50, 'E_cost')[0]:.1f}  "
          f"averaged (5 runs) rho {av5['rho']:+.2f}")
    cases = [
        ("pendulum MLP", Bench(f"{OUT}/baseline_mlp_no_ie.pt", hidden=(16, 16), arch="mlp"),
         "mlp", range(3)),
        ("pendulum GRU-16", Bench(f"{OUT}/baseline_gru16.pt", hidden=16), "gru16", range(3)),
        ("mountain car", MC.bench(), "mct_lam0.003", range(5)),
    ]
    for label, bench, prefix, seeds in cases:
        recs = runs(prefix, seeds)
        sig = load([f"{OUT}/sigma/{prefix}_s{s}.pt" for s in seeds])
        av = averaged(bench, sig)
        rho, rs = ms(recs, "rho")
        print(f"  {label:<16} floor {bench.floor:.1f}  rho {rho:+.2f}+-{rs:.2f}  fall "
              f"{ms(recs, 'fell', 100)[0]:.2f}%  E[J] {ms(recs, 'E_cost')[0]:.1f}  "
              f"averaged ({len(recs)} runs) rho {av['rho']:+.2f} fall {100 * av['fell']:.3f}% "
              f"E[J] {av['E_cost']:.2f} | seed-to-seed agreement {agreement(sig):.2f}")

    print("Section 5.2 (averaging the runs of the first submission)")
    for n in (5, 10, 50):
        r = averaged(b, sig50[:n])
        print(f"  average of {n:>2} runs: rho {r['rho']:+.2f}  fall {100 * r['fell']:.3f}%  "
              f"E[J] {r['E_cost']:.2f}")
    print(f"  seed-to-seed agreement, first 10 runs: {agreement(sig50[:10]):.2f}")

    print("Table 5 (three optimisers of the same objective)")
    F = np.array([neg_elbo(r) for r in main50])
    print(f"  GA, single run (50 runs)      F {F.mean():.1f}+-{F.std():.1f}  rho "
          f"{ms(main50, 'rho')[0]:+.2f}  fall {ms(main50, 'fell', 100)[0]:.2f}%  "
          f"sigma most/least critical decile {ms(main50, 'sigma_most')[0]:.4f}/"
          f"{ms(main50, 'sigma_least')[0]:.3f}")
    print(f"  GA, average of first five     F {av5['neg_elbo']:.1f}  rho {av5['rho']:+.2f}  "
          f"fall {100 * av5['fell']:.2f}%")
    g = json.load(open(f"{OUT}/gradvi_sweep.json"))
    cfg = [c for c in g["configs"] if c["tag"] == "gradvi_K16_lr0.003_clip1"][0]
    best = [r["best"] for r in cfg["runs"]]
    Fg = np.array([r["neg_elbo"] for r in best])
    print(f"  gradient-based VI (5 seeds)   F {Fg.mean():.1f}+-{Fg.std():.1f}  rho "
          f"{np.mean([r['rho'] for r in best]):+.2f}  fall "
          f"{100 * np.mean([r['fell'] for r in best]):.2f}%  sigma most/least critical decile "
          f"{np.mean([r['sigma_most'] for r in best]):.4f}/"
          f"{np.mean([r['sigma_least'] for r in best]):.3f}  rollouts "
          f"{cfg['rollouts_per_seed']:.2e}")
    sg = load([f"{OUT}/sigma/gradvi_K16_lr0.003_clip1_s{s}_best.pt" for s in range(5)])
    geo50 = np.exp(np.log(sig50).mean(0))
    print(f"  rank agreement: gradient seed 0 vs average of the 50 GA runs "
          f"{spearman(sg[0], geo50):.2f} | gradient seed-to-seed {agreement(sg):.3f} | "
          f"GA seed-to-seed (first 10) {agreement(sig50[:10]):.2f}")


if __name__ == "__main__":
    main()
