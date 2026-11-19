# Per-Weight Uncertainty for a Neurocontroller via Evolutionary Variational Inference

Code, trained models, and results for the paper of the same title.

A GRU neurocontroller is trained by an island genetic algorithm on set-point tracking
tasks built on the dynamics of gymnasium **Pendulum-v1** and **MountainCarContinuous-v0**.
A mean-field Gaussian posterior is then evolved over its weights by minimising the
negative ELBO, which yields a per-weight *importance map*: near-zero standard deviation on
weights critical to control, large standard deviation on the rest. The resulting
predictive uncertainty is calibrated to control risk and, on the pendulum, detects the
plant's capability boundary before the mean controller fails. **CartPole-v1** is included
as the contrast case where no such posterior exists around a fixed mean.

The repository holds two sets of experiments. The scripts and results of the first
submission are unchanged (`train_*.py`, `robustness_seeds.py`, `results_pd/`,
`results_cp/`). The experiments added in the revision are in `rev_*.py` and
`results_rev/`.

## Layout

```
ga_torch/                  GA framework + environments
  island.py                island-model GA (TensorBoard logging)
  operators.py             selection / crossover / mutation (incl. log-normal muta_log)
  reproduction.py          configurable breeding plans (default vs self-adaptive)
  pendulum.py              Pendulum-v1 dynamics (batched torch port, validated), task,
                           GRU and MLP controllers, MC evaluators, weight criticality
  mountaincar.py           MountainCarContinuous-v0 dynamics (batched torch port, validated)
  mountaincar_track.py     tracking task on the mountain-car dynamics, same interface as
                           pendulum.py
  cartpole.py              CartPole-v1 via native vectorised gymnasium env
  bayesian.py              gene split, closed-form KL, MC cost clipping

first submission
  train_pendulum.py          deterministic baseline + input ablation   (Table 1, Fig 2)
  train_sigma_pendulum.py    sigma-only ELBO evolution                 (Fig 3)
  robustness_seeds.py        50 sigma-seeds + two further baselines    (Sec. 5.2)
  perturbation_pendulum.py   calibration with 64 samples (superseded by rev_calibration.py)
  hold_sweep.py              boundary sweep with 64 samples (superseded by rev_calibration.py)
  make_figures.py            Figs 2-4
  train_cartpole.py          CartPole baseline + ablations             (Sec. 7.4)
  train_sigma_cartpole.py    CartPole sigma evolution (the negative result)

revision
  rev_pendulum.py            85 runs on the pendulum: lambda sweep, operators, prior,
                             bound, samples, population, generations   (Tables 2-3, Fig 7)
  rev_analysis.py            all first-submission runs re-scored with 2048 samples;
                             sensitivity of the criticality measure    (Sec. 5.2, 6.2)
  rev_calibration.py         calibration and boundary tests with 2048 samples, both
                             plants                                    (Figs 5, 6, 8c)
  rev_arch.py                MLP and larger GRU controllers            (Table 4)
  rev_mc_track.py            mountain car: baseline and sigma evolution (Table 4, Fig 8)
  rev_gradvi.py              gradient-based variational inference      (Table 5)
  rev_timing.py              wall-clock measurements                   (Sec. 6.3)
  rev_tables.py              Tables 2-5 and the averaged runs, recomputed from the
                             stored results (no training)
  rev_figures.py             Figs 5-8 and the graphical abstract
  figures_src/fig_schematic.tex   Fig 1 (TikZ)

results_pd/                first submission, pendulum: trained models (.pt), metrics (.json)
results_cp/                first submission, CartPole
results_rev/               revision: baselines, runs/<tag>.json (one record per run),
                           sigma/<tag>.pt (the evolved sigma of every run), analysis files
```

The figure scripts write to `paper/figures/`, which they create. The manuscript itself is
not part of this repository.

## Setup

Python 3.10+ with PyTorch (CUDA recommended; CPU works but training is slower):

```bash
pip install -r requirements.txt
```

## Quick start: reproduce the figures (no training, about 2 min)

All trained models and measurements are shipped, so the figures regenerate directly:

```bash
python make_figures.py          # Figs 2-4  -> paper/figures/
python rev_figures.py           # Figs 5-8 and the graphical abstract
python rev_pendulum.py --summary    # table of all revision runs
python rev_tables.py            # numbers of Tables 2-5 and of the averaged runs
```

`rev_analysis.py` and `rev_calibration.py` use fixed random seeds; rerunning them
reproduces `results_rev/analysis.json` and `results_rev/calibration_K2048.json` exactly.

## Verify the ports of the dynamics

Gymnasium ships no vectorised Pendulum or MountainCarContinuous, so `ga_torch` implements
the identical update rules batched in torch. To check them against the reference
implementations:

```bash
python -c "from ga_torch.pendulum import validate_against_native as v; print(v(3000))"
# expected: 0.0
python -c "from ga_torch.mountaincar import validate_against_native as v; print(v(40))"
# expected: same termination step and identical return in all episodes,
#           state deviation about 2e-5 (float32 rounding over up to 999 steps)
```

`rev_mc_track.py baseline` additionally runs the trained controller in the real gymnasium
environment and reports the largest position difference to the port (about 1e-7).

## Full reproduction

Runtimes on one NVIDIA RTX 4070 Ti with nothing else on the GPU.

| script | produces | runtime |
|---|---|---|
| `python train_pendulum.py` | baseline + input ablation (Table 1, Fig 2) | ~5 min |
| `python train_sigma_pendulum.py --lams 0.003 --seed 0` | importance map, seed 0 (Fig 3) | ~4 min |
| `python robustness_seeds.py` | 50 sigma-seeds + two further baselines (Sec. 5.2) | ~4 h |
| `python rev_pendulum.py` | lambda sweep, operators, prior, bound, samples, population, generations: 85 runs (Tables 2-3, Fig 7) | ~5 h |
| `python rev_analysis.py` | re-scoring with 2048 samples, criticality sensitivity | ~1 min |
| `python rev_calibration.py` | calibration and boundary tests, both plants (Figs 5, 6, 8c) | ~1 min |
| `python rev_arch.py --mlp --seeds 3` | MLP controller (Table 4) | ~15 min |
| `python rev_arch.py --hidden 16 --seeds 3` | GRU-16 controller (Table 4) | ~25 min |
| `python rev_arch.py --hidden 16 --seeds 1 --gens-sigma 2400 --out-tag _g2400` | one GRU-16 run with three times the generations | ~25 min |
| `python rev_mc_track.py baseline` | mountain-car baseline, two passes for the input scaling | ~5 min |
| `python rev_mc_track.py sigma --lams 0.001,0.003,0.01,0.03 --seeds 0` | mountain car, lambda sweep | ~15 min |
| `python rev_mc_track.py sigma --lams 0.003 --seeds 1,2,3,4` | mountain car, further seeds (Table 4) | ~15 min |
| `python rev_mc_track.py sigma --lams 0.003 --seeds 0,1,2 --plan default` | mountain car, standard operators | ~12 min |
| `python rev_gradvi.py --sweep` | gradient-based VI, 12 settings x 5 seeds (Table 5) | ~30 min |
| `python rev_timing.py` | wall-clock measurements (Sec. 6.3) | ~3 min |
| `python train_cartpole.py` | CartPole baseline + ablations (Sec. 7.4) | ~45 min |
| `python train_sigma_cartpole.py --inputs all6 --lam 1.0` | CartPole negative result | ~90 min |

Caching: the training scripts skip work whose output exists. `train_pendulum.py`,
`train_cartpole.py`, `rev_arch.py` and `rev_mc_track.py` reuse a stored baseline (`.pt`);
`rev_pendulum.py`, `rev_arch.py` and `rev_mc_track.py` skip every run whose record in
`results_rev/runs/` exists. Delete the files to retrain from scratch. Training uses
stochastic evaluation (random set-point sequences and initial states), so retrained numbers
vary slightly around the reported ones; the shipped `.pt` and `.json` files are the exact
artifacts behind the paper.

Two protocol details matter when modifying the code (Sec. 3 of the paper): fitness is
evaluated on freshly drawn random references each generation (a fixed reference lets the
recurrent controller memorise the set-point timetable), and controllers are selected by
re-ranking the final population on the fixed validation battery
(`pendulum.validation_battery`), never by the GA's "best ever" bookkeeping, which is
biased under randomised evaluation.

TensorBoard training curves are written next to the results (`results_*/*_log/`, ignored
by git): `tensorboard --logdir results_rev`.

## License

MIT (see `LICENSE`).
