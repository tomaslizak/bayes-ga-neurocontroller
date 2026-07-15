# Per-Weight Uncertainty for a Neurocontroller via Evolutionary Variational Inference

Code, trained models, and results for the paper (see `paper/main.pdf`).

A GRU neurocontroller is trained by an island genetic algorithm on gymnasium
**Pendulum-v1** (angle set-point tracking near upright). A mean-field Gaussian posterior
is then evolved over its weights by minimising the negative ELBO, which yields a
per-weight *importance map*: near-zero standard deviation on weights critical to control,
large standard deviation on the rest. The resulting predictive uncertainty is calibrated
to control risk and detects the plant's capability boundary before the mean controller
fails. **CartPole-v1** is included as the contrast case where no such posterior exists
around a fixed mean.

## Layout

```
ga_torch/                  GA framework + environments
  island.py                island-model GA (TensorBoard logging)
  operators.py             selection / crossover / mutation (incl. log-normal muta_log)
  reproduction.py          configurable breeding plans (default vs self-adaptive)
  pendulum.py              Pendulum-v1 dynamics (batched torch port, validated), task,
                           MC evaluators, weight criticality
  cartpole.py              CartPole-v1 via native vectorised gymnasium env
  bayesian.py              gene split, closed-form KL, MC cost clipping
train_pendulum.py          deterministic baseline + input ablation   (Table 1, Fig 1)
train_sigma_pendulum.py    sigma-only ELBO evolution                 (Table 2, Fig 2)
perturbation_pendulum.py   physics-perturbation calibration          (Fig 4)
make_figures.py            regenerates all paper figures from results_pd/
train_cartpole.py          CartPole baseline + ablations             (Sec. 4.6)
train_sigma_cartpole.py    CartPole sigma evolution (the negative result)
results_pd/                Pendulum results: trained models (.pt) and metrics (.json)
results_cp/                CartPole results backing Sec. 4.6
paper/                     LaTeX source, figures, compiled PDF
```

## Setup

Python 3.10+ with PyTorch (CUDA recommended; CPU works but training is slower):

```bash
pip install -r requirements.txt
```

## Quick start: reproduce the figures (no training, ~1 min)

All trained models and measurements are shipped, so the paper's figures regenerate
directly:

```bash
python make_figures.py          # -> paper/figures/*.pdf
cd paper && latexmk -pdf main.tex
```

## Verify the Pendulum dynamics port

Gymnasium ships no vectorised Pendulum, so `ga_torch/pendulum.py` implements the identical
update rule batched in torch. To check it against the reference implementation:

```bash
python -c "from ga_torch.pendulum import validate_against_native as v; print('max deviation:', v(3000))"
# expected: 0.0
```

## Full reproduction

Approximate runtimes on a single mid-range CUDA GPU.

| script | produces | runtime |
|---|---|---|
| `python train_pendulum.py` | baseline + input ablation (Table 1, Fig 1) | ~5 min |
| `python train_sigma_pendulum.py --lams 0.003 --seed 0` | importance map, seed 0 (Fig 2) | ~5 min |
| `... --seed 1` / `--seed 2` | reproducibility (Sec. 4.2) | ~5 min each |
| `... --lams 0.003 --seed 0 --plan default` | standard-operator control (Table 2) | ~5 min |
| `python robustness_seeds.py` | 50 sigma-seeds + 3 independent baselines (Sec. 4.2) | ~250 min |
| `... --lams 0.001,0.01,0.03,0.1 --seed 0` | lambda sweep (Sec. 3) | ~20 min |
| `python perturbation_pendulum.py --lam 0.003 --seed 0` | calibration (Fig 4) | ~2 min |
| `python hold_sweep.py` | capability boundary (Fig 5) | ~1 min |
| `python make_figures.py` | all figures (Figs 1-5) | ~1 min |
| `python train_cartpole.py` | CartPole baseline + ablations (Sec. 4.6) | ~45 min |
| `python train_sigma_cartpole.py --inputs all6 --lam 1.0` | CartPole negative result | ~90 min |

Trained-model caching: `train_pendulum.py` and `train_cartpole.py` skip training when the
corresponding `results_*/baseline_*_best.pt` exists; delete those files to retrain from
scratch. Training uses stochastic evaluation (random set-point sequences and initial
states), so retrained numbers vary slightly around the reported ones; the shipped `.pt`
and `.json` files are the exact artifacts behind the paper.

Two protocol details matter when modifying the code (Sec. 2 of the paper): fitness is
evaluated on freshly drawn random references each generation (a fixed reference lets the
recurrent controller memorise the set-point timetable), and controllers are selected by
re-ranking the final population on the fixed validation battery
(`pendulum.validation_battery`), never by the GA's "best ever" bookkeeping, which is
biased under randomised evaluation.

TensorBoard training curves are written next to the results
(`results_pd/*_log/`, ignored by git): `tensorboard --logdir results_pd`.

## License

MIT (see `LICENSE`).
