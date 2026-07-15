"""Per-weight Bayesian GRU controller via evolutionary variational inference.

Built directly on the validated deterministic foundation in `gru_control.py` -- it reuses
the exact same plant, controller and fitness, and only adds the uncertainty layer.

Each GRU weight w_i gets a mean-field Gaussian posterior q(w_i) = N(mu_i, sigma_i^2).
A Bayesian individual is the flat gene vector [mu (n_w) | sigma (n_w)]. The GA minimises
the negative ELBO

    fitness = E_{W ~ q}[ cost(W) ]  +  lam * KL( q || N(0, sigma_prior^2 I) )

with the data term estimated by K-sample Monte-Carlo on `gru_control.simulate`.

Design (see README / discussion):
  * Prior is BROAD (sigma_prior ~ 1) so the KL pulls every sigma_i *up* toward its allowed
    maximum by default -- "big variation everywhere".
  * sigma is bounded to a SMALL range [sigma_min, sigma_max] purely for stability, so weight
    sampling perturbs without destabilising the closed loop.
  * The data term pushes sigma_i *down* only on weights whose perturbation hurts the cost.
  * Net expectation: sigma becomes a per-weight importance map (small = critical, large =
    irrelevant). `weight_criticality()` measures the criticality directly so the prediction
    can be tested instead of assumed.
"""

import math

import torch
from torch import Tensor

from . import gru_control as C
from .population import get_space


# ------------------------------------------------------------------ #
#  Gene geometry
# ------------------------------------------------------------------ #

def gene_count(n_inputs: int, hidden_size: int) -> int:
    """Length of a Bayesian gene vector = 2 * (deterministic weight count)."""
    return 2 * C.gene_count(n_inputs, hidden_size)


def split(genes: Tensor) -> tuple[Tensor, Tensor]:
    """Split (B, 2*n_w) -> (mu, sigma), each (B, n_w)."""
    n_w = genes.shape[1] // 2
    return genes[:, :n_w], genes[:, n_w:]


def make_space(n_inputs: int, hidden_size: int,
               sigma_min: float = 1e-3, sigma_max: float = 0.3,
               device: torch.device = None) -> Tensor:
    """Per-gene GA bounds: mu in [-1, 1], sigma in [sigma_min, sigma_max]."""
    n_w = C.gene_count(n_inputs, hidden_size)
    space = get_space(2 * n_w, -1.0, 1.0, device=device)
    space[0, n_w:] = sigma_min
    space[1, n_w:] = sigma_max
    return space


# ------------------------------------------------------------------ #
#  Closed-form KL  (sum over weights)
# ------------------------------------------------------------------ #

def kl_divergence(mu: Tensor, sigma: Tensor, sigma_prior: float) -> Tensor:
    """KL( N(mu, diag sigma^2) || N(0, sigma_prior^2 I) ), summed over weights -> (B,).

    Per weight: log(sigma_prior/sigma) + (sigma^2 + mu^2)/(2 sigma_prior^2) - 1/2.
    """
    sp2 = sigma_prior * sigma_prior
    per = (math.log(sigma_prior) - torch.log(sigma)
           + (sigma * sigma + mu * mu) / (2.0 * sp2) - 0.5)
    return per.sum(dim=1)


# ------------------------------------------------------------------ #
#  Monte-Carlo evaluation
# ------------------------------------------------------------------ #

# Per-sample cost ceiling for the MC mean. A diverged sample would otherwise contribute the
# 1e8 penalty, making E[cost] a heavy-tailed estimator that a noisy GA exploits via "lucky"
# low-variance draws. Clipping to a finite ceiling makes E[cost] ~= (1-b)*good + b*ceiling,
# which rises smoothly with the divergence fraction b -- a stable, optimisable signal.
COST_CLIP = 1e5


@torch.no_grad()
def simulate_mc(genes: Tensor, ref: Tensor, inputs: tuple[str, ...], scale: Tensor,
                hidden_size: int = 8, K: int = 8, cost_clip: float = COST_CLIP,
                generator: torch.Generator = None, plant_overrides: dict = None) -> dict:
    """K-sample MC rollout of a Bayesian population on the plant.

    Per-sample cost is clipped to `cost_clip` before averaging (see COST_CLIP).
    `plant_overrides` perturbs the plant (for robustness tests); None = nominal.

    Returns (B,) tensors: expected_cost, cost_std, blown_rate (fraction of the K samples
    that diverged) plus mu, sigma views.
    """
    mu, sigma = split(genes)
    B, n_w = mu.shape
    eps = torch.randn(B, K, n_w, device=genes.device, dtype=genes.dtype, generator=generator)
    W = (mu.unsqueeze(1) + sigma.unsqueeze(1) * eps).reshape(B * K, n_w)

    sim = C.simulate(W, ref, inputs, scale, hidden_size, plant_overrides=plant_overrides)
    cost = C.fitness(sim).clamp(max=cost_clip).reshape(B, K)
    blown_rate = sim["blown"].reshape(B, K).float().mean(dim=1)

    return {"expected_cost": cost.mean(dim=1), "cost_std": cost.std(dim=1),
            "blown_rate": blown_rate, "mu": mu, "sigma": sigma}


def make_evaluator(ref: Tensor, inputs: tuple[str, ...], scale: Tensor,
                   hidden_size: int, K: int, lam: float, sigma_prior: float,
                   cost_clip: float = COST_CLIP):
    """Build the (minimising) negative-ELBO evaluator for IslandGA (full mu+sigma genes)."""
    def evaluate(genes: Tensor) -> Tensor:
        out = simulate_mc(genes, ref, inputs, scale, hidden_size, K, cost_clip)
        return out["expected_cost"] + lam * kl_divergence(out["mu"], out["sigma"], sigma_prior)
    return evaluate


# ------------------------------------------------------------------ #
#  Stage A: evolve ONLY sigma, with mu fixed at a trained controller
# ------------------------------------------------------------------ #

def sigma_space(n_inputs: int, hidden_size: int,
                sigma_min: float = 1e-3, sigma_max: float = 0.3,
                device: torch.device = None) -> Tensor:
    """GA bounds for a sigma-only genome (n_w genes, all in [sigma_min, sigma_max])."""
    return get_space(C.gene_count(n_inputs, hidden_size), sigma_min, sigma_max, device=device)


def make_sigma_only_evaluator(mu_fixed: Tensor, ref: Tensor, inputs: tuple[str, ...],
                              scale: Tensor, hidden_size: int, K: int,
                              lam: float, sigma_prior: float, cost_clip: float = COST_CLIP):
    """Negative-ELBO evaluator where the genome is sigma only; mu is held fixed.

    `mu_fixed` is a (n_w,) trained controller. The GA population is (B, n_w) of sigmas.
    """
    mu_row = mu_fixed.reshape(1, -1)
    def evaluate(sigmas: Tensor) -> Tensor:
        genes = torch.cat([mu_row.expand(sigmas.shape[0], -1), sigmas], dim=1)
        out = simulate_mc(genes, ref, inputs, scale, hidden_size, K, cost_clip)
        return out["expected_cost"] + lam * kl_divergence(out["mu"], out["sigma"], sigma_prior)
    return evaluate


# ------------------------------------------------------------------ #
#  Diagnostic: per-weight criticality (to test the importance-map prediction)
# ------------------------------------------------------------------ #

@torch.no_grad()
def weight_criticality(mu: Tensor, ref: Tensor, inputs: tuple[str, ...], scale: Tensor,
                       hidden_size: int = 8, delta: float = 0.05) -> Tensor:
    """How much each weight matters: mean |Δcost| when w_i is nudged by ±delta.

    `mu` is a single (n_w,) controller (the posterior mean). All 2*n_w perturbations are
    simulated in one batched call. Returns (n_w,) non-negative criticality.

    The Bayesian prediction is: evolved sigma_i should be SMALL where criticality is high
    and LARGE where it is low (a negative sigma-vs-criticality correlation).
    """
    n_w = mu.shape[0]
    base = C.fitness(C.simulate(mu.unsqueeze(0), ref, inputs, scale, hidden_size))[0]
    eye = delta * torch.eye(n_w, device=mu.device, dtype=mu.dtype)
    pert = torch.cat([mu.unsqueeze(0) + eye, mu.unsqueeze(0) - eye], dim=0)  # (2n_w, n_w)
    cost = C.fitness(C.simulate(pert, ref, inputs, scale, hidden_size))
    cp, cm = cost[:n_w], cost[n_w:]
    return 0.5 * ((cp - base).abs() + (cm - base).abs())
