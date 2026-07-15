"""GRU angle-tracking controller on gymnasium Pendulum-v1 dynamics (batched, validated).

The benchmark is gymnasium's Pendulum-v1: continuous torque u in [-2, 2], no termination,
dt = 0.05, theta = 0 upright. Gymnasium ships no vectorised Pendulum (SyncVectorEnv would
cost ~minutes per GA generation), so this module implements the identical update rule

    thdot' = thdot + (3 g / (2 l) sin(th) + 3 / (m l^2) u) dt,  clipped to +-max_speed
    th'    = th + thdot' dt

batched in torch, and `validate_against_native()` checks it against the reference
gymnasium implementation step by step (float64, deviation ~1e-12 over thousands of steps).

Task (ours, on top of the native dynamics): track a piecewise-constant ANGLE set-point
near upright. Set-points stay inside +-0.35 rad, within the statically holdable range
sin(th) <= 2 u_max / (m g l) = 0.4. The GRU outputs u = max_torque * tanh(.), the native
continuous action. Episodes start near upright (theta ~ U(-0.1, 0.1), thdot ~ U(-0.5, 0.5)).

Cost (the GA minimises):  J = sum_k |e_k| + alfa * sum_k |th_k - th_{k-1}|,
with e the angle-wrapped tracking error. There is no termination, so no death penalty and
no cost cliff; a pendulum that swings away pays through |e| itself. The `fell` diagnostic
marks episodes that ever left the upper half plane (|theta| > pi/2).

Candidate controller inputs: e, ie (integral of e), theta, theta_dot. Same 1/p99 scaling
rule and the same anti-memorisation protocol as the CartPole port: random references per
evaluation, a fixed validation battery for selection, a held-out canonical reference for
figures.
"""

import numpy as np
import torch
from torch import Tensor

from .simulation import unpack_gru_weights

# ------------------------------------------------------------------ #
#  Native physics constants (gymnasium PendulumEnv defaults) and task
# ------------------------------------------------------------------ #

PHYS = dict(g=10.0, m=1.0, l=1.0, max_torque=2.0, max_speed=8.0)
DT = 0.05
NSTEPS = 440                    # 22 s: 2 s settle + 4 x 5 s set-point segments
ALFA = 10.0                     # smoothness weight on |d theta|
# Set-point magnitude cap. Statically holdable angle is asin(2 u_max / (m g l)) = 0.41 rad;
# 0.30 keeps a ~26% torque margin at the nominal plant so that nominal operation is
# comfortably feasible and predictive uncertainty at nominal stays low. (An earlier 0.35
# cap put the largest set-points at 86% of the limit, inflating nominal uncertainty.)
REF_MAX = 0.30

ALL_FEATURES = ("e", "ie", "theta", "theta_dot")

# Per-signal scales, 1/p99 rule, measured by train_pendulum.py from the trained no_ie
# controller on the held-out reference; see results_pd/baseline_compare.json.
DEFAULT_SCALE = {"e": 1/0.51, "ie": 1/0.21, "theta": 1/0.30, "theta_dot": 1/0.93}

PHYS_PARAMS = ("g", "m", "l", "max_torque")


def wrap(a):
    """Wrap angle(s) to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def get_reference(nsteps: int = NSTEPS, device: torch.device = None) -> Tensor:
    """FIXED held-out evaluation set-point sequence: 0 -> 0.25 -> -0.20 -> 0.12 -> 0 rad.

    Levels are TYPICAL of the training distribution (|level| <= 0.83 * REF_MAX), not its
    extreme: behaviour exactly at the REF_MAX boundary is studied separately by the
    hold-level sweep, where the posterior's uncertainty is shown to blow up as the
    set-point approaches the static torque limit.
    """
    t = torch.arange(nsteps, device=device, dtype=torch.float32) * DT
    r = torch.zeros(nsteps, device=device, dtype=torch.float32)
    r[t >= 2] = 0.25
    r[t >= 7] = -0.20
    r[t >= 12] = 0.12
    r[t >= 17] = 0.0
    return r


def get_random_reference(nsteps: int = NSTEPS, rng: np.random.Generator = None,
                         device: torch.device = None) -> Tensor:
    """Random training sequence: 2 s at 0, then four 5 s levels ~ U(-REF_MAX, REF_MAX)."""
    rng = rng or np.random.default_rng()
    levels = rng.uniform(-REF_MAX, REF_MAX, size=4)
    t = torch.arange(nsteps, device=device, dtype=torch.float32) * DT
    r = torch.zeros(nsteps, device=device, dtype=torch.float32)
    for i, lv in enumerate(levels):
        r[t >= 2 + 5 * i] = float(lv)
    return r


def validation_battery(n_refs: int = 8, nsteps: int = NSTEPS, seed: int = 777,
                       device: torch.device = None) -> Tensor:
    """Fixed battery of random references for selecting/ranking (see cartpole.py notes)."""
    rng = np.random.default_rng(seed)
    return torch.stack([get_random_reference(nsteps, rng, device) for _ in range(n_refs)])


def gene_count(n_inputs: int, hidden_size: int) -> int:
    return 3 * hidden_size * n_inputs + 3 * hidden_size * hidden_size + hidden_size


def scale_tensor(inputs: tuple[str, ...], scale_map: dict = None,
                 device: torch.device = None, dtype: torch.dtype = torch.float32) -> Tensor:
    scale_map = scale_map or DEFAULT_SCALE
    return torch.tensor([scale_map[n] for n in inputs], device=device, dtype=dtype)


def make_init_states(n_episodes: int, seed: int = 0) -> np.ndarray:
    """Reproducible near-upright initial states: (E, 2) of [theta, theta_dot]."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(-0.1, 0.1, size=n_episodes)
    thd = rng.uniform(-0.5, 0.5, size=n_episodes)
    return np.stack([th, thd], axis=1)


# ------------------------------------------------------------------ #
#  GRU step (compiled) -- continuous torque output
# ------------------------------------------------------------------ #

@torch.compile
def _gru_step(x_t: Tensor, h: Tensor, W_z, W_r, W_h, U_z, U_r, U_h, W_out,
              max_torque: float) -> tuple[Tensor, Tensor]:
    z = torch.sigmoid(torch.bmm(W_z, x_t.unsqueeze(2)).squeeze(2)
                      + torch.bmm(U_z, h.unsqueeze(2)).squeeze(2))
    r = torch.sigmoid(torch.bmm(W_r, x_t.unsqueeze(2)).squeeze(2)
                      + torch.bmm(U_r, h.unsqueeze(2)).squeeze(2))
    h_tilde = torch.tanh(torch.bmm(W_h, x_t.unsqueeze(2)).squeeze(2)
                         + torch.bmm(U_h, (r * h).unsqueeze(2)).squeeze(2))
    h_new = (1.0 - z) * h + z * h_tilde
    u = torch.tanh((h_new * W_out).sum(dim=1)) * max_torque
    return h_new, u


# ------------------------------------------------------------------ #
#  Batched closed-loop rollout (pure torch, GPU end to end)
# ------------------------------------------------------------------ #

@torch.no_grad()
def simulate(genes: Tensor, ref: Tensor, inputs: tuple[str, ...], scale: Tensor,
             hidden_size: int = 8, episodes: int = 1,
             init_seed: int = None, plant_overrides: dict = None,
             return_features: bool = False, return_traj: bool = False) -> dict:
    """Batched closed-loop rollout; mirrors cartpole.simulate (see that docstring).

    ref: (nsteps,) shared, or (episodes, nsteps) one row per episode slot.
    Returns dict with cost (B,), fell_rate (B,), and optional features / theta traj.
    """
    p = dict(PHYS)
    if plant_overrides:
        for k, v in plant_overrides.items():
            if k not in PHYS_PARAMS:
                raise ValueError(f"unknown physics parameter '{k}'")
            p[k] = v
    g, m, l = p["g"], p["m"], p["l"]
    max_torque, max_speed = p["max_torque"], p["max_speed"]

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

    # initial state
    if init_seed is not None:
        st = make_init_states(E, init_seed)                       # (E, 2)
        st = np.tile(st, (B, 1))                                  # (N, 2), episode e = i % E
        th = torch.tensor(st[:, 0], device=device, dtype=dtype)
        thdot = torch.tensor(st[:, 1], device=device, dtype=dtype)
    else:
        th = (torch.rand(N, device=device, dtype=dtype) - 0.5) * 0.2
        thdot = (torch.rand(N, device=device, dtype=dtype) - 0.5) * 1.0

    h = torch.zeros(N, hidden_size, device=device, dtype=dtype)
    ie = torch.zeros(N, device=device, dtype=dtype)
    cost = torch.zeros(N, device=device, dtype=dtype)
    fell = torch.zeros(N, device=device, dtype=torch.bool)
    th_prev = th.clone()
    two_pi = 2 * np.pi

    feats = ({f: torch.zeros(N, nsteps, device=device, dtype=dtype) for f in ALL_FEATURES}
             if return_features else None)
    th_traj = torch.zeros(N, nsteps, device=device, dtype=dtype) if return_traj else None
    u_traj = torch.zeros(N, nsteps, device=device, dtype=dtype) if return_traj else None

    for k in range(nsteps):
        r_k = ref_t[k] if ref_of_env is None else ref_of_env[:, k]
        th_w = (th + np.pi) % two_pi - np.pi                      # wrapped angle
        e = (r_k - th + np.pi) % two_pi - np.pi                   # wrapped error
        ie = ie + e * DT
        fell = fell | (th_w.abs() > np.pi / 2)

        cur = {"e": e, "ie": ie, "theta": th_w, "theta_dot": thdot}
        if return_features:
            for f in ALL_FEATURES:
                feats[f][:, k] = cur[f]

        cost = cost + e.abs() + ALFA * (th - th_prev).abs()
        th_prev = th.clone()

        x_t = torch.stack([cur[n] for n in inputs], dim=1) * scale
        h, u = _gru_step(x_t, h, *W, max_torque)
        if return_traj:
            th_traj[:, k] = th_w
            u_traj[:, k] = u

        # native Pendulum-v1 update rule
        thdot = thdot + (3.0 * g / (2.0 * l) * torch.sin(th) + 3.0 / (m * l * l) * u) * DT
        thdot = torch.clamp(thdot, -max_speed, max_speed)
        th = th + thdot * DT

    out = {"cost": cost.reshape(B, E).mean(dim=1),
           "fell_rate": fell.reshape(B, E).float().mean(dim=1)}
    if return_features:
        out["features"] = {f: v for f, v in feats.items()}
    if return_traj:
        out["theta"] = th_traj
        out["u"] = u_traj
    return out


# ------------------------------------------------------------------ #
#  Validation against the native gymnasium implementation
# ------------------------------------------------------------------ #

def validate_against_native(n_steps: int = 3000, seed: int = 0) -> float:
    """Step the native Pendulum-v1 and our update rule side by side with identical random
    torques; return the maximum absolute deviation in (theta, thetadot). Should be ~1e-12.
    """
    import gymnasium as gym
    rng = np.random.default_rng(seed)
    env = gym.make("Pendulum-v1")
    env.reset(seed=seed)
    u_env = env.unwrapped
    th0, thd0 = 0.07, -0.3
    u_env.state = np.array([th0, thd0])

    th, thd = np.float64(th0), np.float64(thd0)
    p = PHYS
    worst = 0.0
    for _ in range(n_steps):
        a = rng.uniform(-2.0, 2.0)
        env.step(np.array([a], dtype=np.float64))
        u = np.clip(a, -p["max_torque"], p["max_torque"])
        thd = thd + (3 * p["g"] / (2 * p["l"]) * np.sin(th) + 3.0 / (p["m"] * p["l"] ** 2) * u) * DT
        thd = np.clip(thd, -p["max_speed"], p["max_speed"])
        th = th + thd * DT
        worst = max(worst,
                    abs(th - u_env.state[0]), abs(thd - u_env.state[1]))
    env.close()
    return float(worst)


# ------------------------------------------------------------------ #
#  Bayesian evaluators (mirror cartpole.py; generic bits from ga_torch.bayesian)
# ------------------------------------------------------------------ #

from . import bayesian as _bayes                                    # noqa: E402


@torch.no_grad()
def simulate_mc(genes: Tensor, ref: Tensor, inputs: tuple[str, ...], scale: Tensor,
                hidden_size: int = 8, K: int = 8, episodes: int = 1,
                cost_clip: float = _bayes.COST_CLIP, init_seed: int = None,
                plant_overrides: dict = None,
                generator: torch.Generator = None) -> dict:
    """K-sample Monte-Carlo evaluation of Bayesian [mu | sigma] individuals."""
    mu, sigma = _bayes.split(genes)
    B, n_w = mu.shape
    eps = torch.randn(B, K, n_w, device=genes.device, dtype=genes.dtype, generator=generator)
    Ws = (mu.unsqueeze(1) + sigma.unsqueeze(1) * eps).reshape(B * K, n_w)
    out = simulate(Ws, ref, inputs, scale, hidden_size, episodes,
                   init_seed=init_seed, plant_overrides=plant_overrides)
    cost = out["cost"].clamp(max=cost_clip).reshape(B, K)
    return {"expected_cost": cost.mean(dim=1), "cost_std": cost.std(dim=1),
            "fell_rate": out["fell_rate"].reshape(B, K).mean(dim=1),
            "mu": mu, "sigma": sigma}


@torch.no_grad()
def weight_criticality(mu: Tensor, ref: Tensor, inputs: tuple[str, ...], scale: Tensor,
                       hidden_size: int = 8, delta: float = 0.05,
                       episodes: int = 8, init_seed: int = 555) -> Tensor:
    """Per-weight criticality: mean |Delta cost| for +-delta nudges, on fixed initial
    states and the given (battery) reference so all comparisons share the conditions."""
    n_w = mu.shape[0]
    eye = delta * torch.eye(n_w, device=mu.device, dtype=mu.dtype)
    batch = torch.cat([mu.unsqueeze(0), mu.unsqueeze(0) + eye, mu.unsqueeze(0) - eye], dim=0)
    cost = simulate(batch, ref, inputs, scale, hidden_size, episodes,
                    init_seed=init_seed)["cost"]
    base, cp, cm = cost[0], cost[1:n_w + 1], cost[n_w + 1:]
    return 0.5 * ((cp - base).abs() + (cm - base).abs())
