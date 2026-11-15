"""Position tracking on gymnasium MountainCarContinuous-v0 dynamics (second benchmark).

Same construction as the pendulum benchmark: the native state update, with our set-point
tracking task on top. The module exposes the same functions as `pendulum.py` (simulate,
simulate_mc, weight_criticality, references, battery), so the experiment drivers run on
either plant unchanged.

Plant: `mountaincar.plant_step`, the native update rule checked against gymnasium:

    v' = clip(v + a * power - 0.0025 cos(3 x), -0.07, 0.07),   x' = clip(x + v', -1.2, 0.6)

with the native action a in [-1, 1] and power = 0.0015. The car sits in a valley whose
bottom is at x_b = -pi/6 = -0.524. There is no friction, so the open loop is an undamped
oscillator (period about 73 steps near the bottom), and the controller has to damp it.

Task (ours): hold the car at a commanded position on the slope. Holding the offset d from
the bottom needs the steady action a = (0.0025 / 0.0015) sin(3 d), so positions are
holdable only for |d| <= asin(0.6) / 3 = 0.214. Set-points are drawn within
REF_MAX = 0.15 of the bottom, where the steady action is at most 0.725, a 27 % force margin
(the pendulum task leaves 26 % of the torque). The goal flag of the native environment
plays no role: the car is meant to stay in the valley and the episode is not terminated.

Cost (the GA minimises):  J = sum_k |e_k| + alfa * sum_k |x_k - x_{k-1}|,  e = r - x,
the same form and the same alfa = 10 as on the pendulum. The `fell` diagnostic marks
episodes in which the car ever left the valley (reached the goal position or the left
wall).

Episodes are 440 steps: 40 steps at the valley bottom, then four set-point levels of 100
steps each. Initial states follow the native reset, x0 ~ U(-0.6, -0.4), v0 = 0. Protocol
as on the pendulum: random references per evaluation, a fixed validation battery for
selection, a held-out reference for figures.
"""

import numpy as np
import torch
from torch import Tensor

from .simulation import unpack_gru_weights
from .pendulum import _gru_step
from .mountaincar import PHYS, RESET_LOW, RESET_HIGH, plant_step
from . import bayesian as _bayes

DT = 1.0                         # one native step (the environment has no physical time)
NSTEPS = 440                     # 40 steps settle + 4 x 100 steps set-point segments
SETTLE, SEGMENT = 40, 100
ALFA = 10.0
X_BOTTOM = -np.pi / 6            # valley bottom, cos(3 x) = 0
REF_MAX = 0.15                   # largest set-point offset from the bottom
HOLD_LIMIT = float(np.arcsin(PHYS["power"] / PHYS["gravity"]) / 3)      # 0.2145

ALL_FEATURES = ("e", "x", "v")
# Input scales, 1/p99 rule. First pass: rough values from the task geometry; replaced by
# the values measured on a trained controller (rev_mc_track.py prints and stores them).
ROUGH_SCALE = {"e": 1 / 0.30, "x": 1 / 0.70, "v": 1 / 0.07}
DEFAULT_SCALE = dict(ROUGH_SCALE)

PHYS_PARAMS = ("power", "gravity")


def _levels_to_ref(levels, nsteps: int, device) -> Tensor:
    r = torch.full((nsteps,), X_BOTTOM, device=device, dtype=torch.float32)
    for i, lv in enumerate(levels):
        r[SETTLE + SEGMENT * i:] = X_BOTTOM + float(lv)
    return r


def get_reference(nsteps: int = NSTEPS, device: torch.device = None) -> Tensor:
    """FIXED held-out set-point sequence, offsets +0.125 -> -0.10 -> +0.06 -> 0 from the
    bottom (the pendulum's held-out levels scaled by REF_MAX; at most 0.83 * REF_MAX)."""
    return _levels_to_ref([0.125, -0.10, 0.06, 0.0], nsteps, device)


def get_random_reference(nsteps: int = NSTEPS, rng: np.random.Generator = None,
                         device: torch.device = None) -> Tensor:
    """Random training sequence: 40 steps at the bottom, then four levels of 100 steps with
    offsets ~ U(-REF_MAX, REF_MAX)."""
    rng = rng or np.random.default_rng()
    return _levels_to_ref(rng.uniform(-REF_MAX, REF_MAX, size=4), nsteps, device)


def validation_battery(n_refs: int = 8, nsteps: int = NSTEPS, seed: int = 777,
                       device: torch.device = None) -> Tensor:
    rng = np.random.default_rng(seed)
    return torch.stack([get_random_reference(nsteps, rng, device) for _ in range(n_refs)])


def gene_count(n_inputs: int, hidden_size: int, arch: str = "gru") -> int:
    return 3 * hidden_size * n_inputs + 3 * hidden_size * hidden_size + hidden_size


def scale_tensor(inputs: tuple[str, ...], scale_map: dict = None,
                 device: torch.device = None, dtype: torch.dtype = torch.float32) -> Tensor:
    scale_map = scale_map or DEFAULT_SCALE
    return torch.tensor([scale_map[n] for n in inputs], device=device, dtype=dtype)


def make_init_states(n_episodes: int, seed: int = 0) -> np.ndarray:
    """Reproducible initial states from the native reset: (E, 2) of [x, v]."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(RESET_LOW, RESET_HIGH, size=n_episodes),
                     np.zeros(n_episodes)], axis=1)


@torch.no_grad()
def simulate(genes: Tensor, ref: Tensor, inputs: tuple[str, ...], scale: Tensor,
             hidden_size: int = 8, episodes: int = 1,
             init_seed: int = None, plant_overrides: dict = None,
             return_features: bool = False, return_traj: bool = False,
             arch: str = "gru") -> dict:
    """Batched closed-loop rollout; same interface and bookkeeping as pendulum.simulate.

    ref: (nsteps,) shared, or (episodes, nsteps) one row per episode slot.
    Returns dict with cost (B,), fell_rate (B,), and optional features / x and u traj.
    """
    if arch != "gru":
        raise ValueError("only the GRU controller is implemented for this plant")
    p = dict(PHYS)
    if plant_overrides:
        for k, val in plant_overrides.items():
            if k not in PHYS_PARAMS:
                raise ValueError(f"unknown physics parameter '{k}'")
            p[k] = val

    device, dtype = genes.device, genes.dtype
    B = genes.shape[0]
    E = episodes
    N = B * E
    nsteps = ref.shape[-1]

    W = unpack_gru_weights(torch.repeat_interleave(genes, E, dim=0), len(inputs), hidden_size)
    scale = scale.to(device=device, dtype=dtype)

    ref_t = ref.to(device=device, dtype=dtype)
    if ref_t.ndim == 2:
        if ref_t.shape[0] != E:
            raise ValueError(f"per-episode ref needs shape ({E}, nsteps), got {tuple(ref_t.shape)}")
        ref_of_env = ref_t[torch.arange(N, device=device) % E]   # (N, nsteps)
    else:
        ref_of_env = None

    if init_seed is not None:
        st = np.tile(make_init_states(E, init_seed), (B, 1))     # episode e = i % E
        x = torch.tensor(st[:, 0], device=device, dtype=dtype)
        v = torch.tensor(st[:, 1], device=device, dtype=dtype)
    else:
        x = RESET_LOW + (RESET_HIGH - RESET_LOW) * torch.rand(N, device=device, dtype=dtype)
        v = torch.zeros(N, device=device, dtype=dtype)

    h = torch.zeros(N, hidden_size, device=device, dtype=dtype)
    cost = torch.zeros(N, device=device, dtype=dtype)
    fell = torch.zeros(N, device=device, dtype=torch.bool)
    x_prev = x.clone()

    feats = ({f: torch.zeros(N, nsteps, device=device, dtype=dtype) for f in ALL_FEATURES}
             if return_features else None)
    x_traj = torch.zeros(N, nsteps, device=device, dtype=dtype) if return_traj else None
    u_traj = torch.zeros(N, nsteps, device=device, dtype=dtype) if return_traj else None

    for k in range(nsteps):
        r_k = ref_t[k] if ref_of_env is None else ref_of_env[:, k]
        e = r_k - x
        fell = fell | (x >= p["goal_position"]) | (x <= p["min_position"])

        cur = {"e": e, "x": x, "v": v}
        if return_features:
            for f in ALL_FEATURES:
                feats[f][:, k] = cur[f]

        cost = cost + e.abs() + ALFA * (x - x_prev).abs()
        x_prev = x.clone()

        x_t = torch.stack([cur[n] for n in inputs], dim=1) * scale
        h, u = _gru_step(x_t, h, *W, 1.0)                 # native action bound
        if return_traj:
            x_traj[:, k] = x
            u_traj[:, k] = u

        x, v, _ = plant_step(x, v, u, p)                  # native update; goal flag unused

    out = {"cost": cost.reshape(B, E).mean(dim=1),
           "fell_rate": fell.reshape(B, E).float().mean(dim=1)}
    if return_features:
        out["features"] = {f: val for f, val in feats.items()}
    if return_traj:
        out["x"] = x_traj
        out["u"] = u_traj
    return out


@torch.no_grad()
def simulate_mc(genes: Tensor, ref: Tensor, inputs: tuple[str, ...], scale: Tensor,
                hidden_size: int = 8, K: int = 8, episodes: int = 1,
                cost_clip: float = _bayes.COST_CLIP, init_seed: int = None,
                plant_overrides: dict = None,
                generator: torch.Generator = None, arch: str = "gru") -> dict:
    """K-sample Monte-Carlo evaluation of Bayesian [mu | sigma] individuals."""
    mu, sigma = _bayes.split(genes)
    B, n_w = mu.shape
    eps = torch.randn(B, K, n_w, device=genes.device, dtype=genes.dtype, generator=generator)
    Ws = (mu.unsqueeze(1) + sigma.unsqueeze(1) * eps).reshape(B * K, n_w)
    out = simulate(Ws, ref, inputs, scale, hidden_size, episodes,
                   init_seed=init_seed, plant_overrides=plant_overrides, arch=arch)
    cost = out["cost"].clamp(max=cost_clip).reshape(B, K)
    return {"expected_cost": cost.mean(dim=1), "cost_std": cost.std(dim=1),
            "fell_rate": out["fell_rate"].reshape(B, K).mean(dim=1),
            "mu": mu, "sigma": sigma}


@torch.no_grad()
def weight_criticality(mu: Tensor, ref: Tensor, inputs: tuple[str, ...], scale: Tensor,
                       hidden_size: int = 8, delta: float = 0.05,
                       episodes: int = 8, init_seed: int = 555,
                       arch: str = "gru") -> Tensor:
    """Per-weight criticality: mean |Delta cost| for +-delta nudges, on fixed initial
    states and the given (battery) reference."""
    n_w = mu.shape[0]
    eye = delta * torch.eye(n_w, device=mu.device, dtype=mu.dtype)
    batch = torch.cat([mu.unsqueeze(0), mu.unsqueeze(0) + eye, mu.unsqueeze(0) - eye], dim=0)
    cost = simulate(batch, ref, inputs, scale, hidden_size, episodes,
                    init_seed=init_seed, arch=arch)["cost"]
    base, cp, cm = cost[0], cost[1:n_w + 1], cost[n_w + 1:]
    return 0.5 * ((cp - base).abs() + (cm - base).abs())
