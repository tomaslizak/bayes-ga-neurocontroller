"""GRU tracking controller on the native gymnasium CartPole-v1.

This is the gymnasium counterpart of :mod:`gru_control`: same GRU controller, same GA and
Bayesian machinery, but the plant is gymnasium's own ``CartPole-v1`` (numpy-vectorised
``CartPoleVectorEnv``), not a custom ODE. We use the environment exactly as shipped --
its physics, its discrete left/right action, its termination rules. Only the *task* is
ours: the cart must track a piecewise-constant position set-point while keeping the pole
up. The GRU outputs a scalar u; the native discrete action is ``1`` if u > 0 else ``0``.

Candidate controller inputs (canonical order):

    e        set-point tracking error      r - x
    ie       integral of error             sum e * tau
    x        cart position                 (native obs 0)
    x_dot    cart velocity                 (native obs 1)
    theta    pole angle                    (native obs 2)
    theta_dot pole angular velocity        (native obs 3)

Each fed input is multiplied by a per-signal scale (the 1/p99 rule, measured from a
trained controller -- same procedure as for the ODE plant).

Cost (the GA minimises):  J = sum_k alive_k * (|e_k| + alfa*|x_k - x_{k-1}|)
                              + sum_k (1 - alive_k) * P
i.e. tracking error + cart-motion smoothness while the pole is up, and a fixed per-step
penalty P for every step after the pole has fallen (or the cart left the track). Falling
early is the analogue of the ODE plant's divergence; ``blown`` marks episodes that died
before the horizon.

Plant perturbations for the robustness experiments are applied through the environment's
own physics attributes (``masspole``, ``length``, ``force_mag``, ...). CartPole caches
``total_mass`` and ``polemass_length`` at construction, so those are recomputed after an
override.
"""

import numpy as np
import torch
from torch import Tensor

import gymnasium as gym

from .simulation import unpack_gru_weights

# ------------------------------------------------------------------ #
#  Task definition
# ------------------------------------------------------------------ #

TAU = 0.02                      # CartPole-v1 integration step [s]
NSTEPS = 1500                   # 30 s horizon
ALFA = 10.0                     # smoothness weight on |dx|
DEAD_PENALTY = 6.0              # per-step cost once the pole has fallen
X_LIMIT = 2.4                   # native track limit (termination)

ALL_FEATURES = ("e", "ie", "x", "x_dot", "theta", "theta_dot")

# Per-signal input scales, 1/p99 rule: measured by train_cartpole.py from the trained
# 1000-generation controller tracking the held-out sequence (survives 4/4, worst error
# 0.031 m); see results_cp/baseline_compare.json.
DEFAULT_SCALE = {"e": 1/2.54, "ie": 1/2.02, "x": 1/1.57,
                 "x_dot": 1/1.64, "theta": 1/0.19, "theta_dot": 1/0.73}

# Physics attributes that may be perturbed via plant_overrides.
PHYS_PARAMS = ("gravity", "masscart", "masspole", "length", "force_mag")


def get_reference(nsteps: int = NSTEPS, device: torch.device = None) -> Tensor:
    """The FIXED evaluation set-point sequence: 0 -> 1.5 -> -1.0 -> 0.5 -> 0.

    Stays well inside the +-2.4 track limit. Segment changes at 2, 9, 16, 23 s.
    Held out from training (training uses get_random_reference), so tracking it
    demonstrates feedback control rather than a memorised schedule.
    """
    t = torch.arange(nsteps, device=device, dtype=torch.float32) * TAU
    r = torch.zeros(nsteps, device=device, dtype=torch.float32)
    r[t >= 2] = 1.5
    r[t >= 9] = -1.0
    r[t >= 16] = 0.5
    r[t >= 23] = 0.0
    return r


def get_random_reference(nsteps: int = NSTEPS, rng: np.random.Generator = None,
                         device: torch.device = None) -> Tensor:
    """A random training set-point sequence: 2 s at 0, then four 7 s levels ~ U(-1.5, 1.5).

    A GRU can exploit a fixed reference by memorising its timetable (anticipating
    set-point changes instead of reacting to the error). Training on a fresh random
    sequence every evaluation removes that shortcut; the fixed get_reference sequence
    then serves as a held-out test.
    """
    rng = rng or np.random.default_rng()
    levels = rng.uniform(-1.5, 1.5, size=4)
    t = torch.arange(nsteps, device=device, dtype=torch.float32) * TAU
    r = torch.zeros(nsteps, device=device, dtype=torch.float32)
    for i, lv in enumerate(levels):
        r[t >= 2 + 7 * i] = float(lv)
    return r


def gene_count(n_inputs: int, hidden_size: int) -> int:
    """Flat gene-vector length of the GRU controller (same layout as gru_control)."""
    return 3 * hidden_size * n_inputs + 3 * hidden_size * hidden_size + hidden_size


def scale_tensor(inputs: tuple[str, ...], scale_map: dict = None,
                 device: torch.device = None, dtype: torch.dtype = torch.float32) -> Tensor:
    scale_map = scale_map or DEFAULT_SCALE
    return torch.tensor([scale_map[n] for n in inputs], device=device, dtype=dtype)


# ------------------------------------------------------------------ #
#  Environment helpers
# ------------------------------------------------------------------ #

def _make_env(num_envs: int, nsteps: int, plant_overrides: dict = None):
    """Create the native vectorised CartPole and apply physics overrides."""
    env = gym.make_vec("CartPole-v1", num_envs=num_envs,
                       vectorization_mode="vector_entry_point",
                       max_episode_steps=nsteps + 2)   # we stop ourselves at nsteps
    u = env.unwrapped
    if plant_overrides:
        for k, v in plant_overrides.items():
            if k not in PHYS_PARAMS:
                raise ValueError(f"unknown physics parameter '{k}'")
            setattr(u, k, v)
        # CartPole caches these two at construction; recompute after overrides.
        u.total_mass = u.masspole + u.masscart
        u.polemass_length = u.masspole * u.length
    return env


def make_init_states(n_episodes: int, seed: int = 0) -> np.ndarray:
    """Reproducible per-episode initial states, uniform(-0.05, 0.05)^4 like the native reset."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.05, 0.05, size=(n_episodes, 4))


def validation_battery(n_refs: int = 8, nsteps: int = NSTEPS, seed: int = 777,
                       device: torch.device = None) -> Tensor:
    """A FIXED set of random set-point sequences, (n_refs, nsteps), for selecting/ranking
    controllers under randomised-reference training.

    Training references change every evaluation, so fitnesses are not comparable across
    generations and "best ever" bookkeeping selects whoever met the easiest reference.
    Rank candidates on this fixed battery (with a fixed init_seed) instead. The canonical
    get_reference sequence stays fully held out for figures/tests.
    """
    rng = np.random.default_rng(seed)
    return torch.stack([get_random_reference(nsteps, rng, device) for _ in range(n_refs)])


# ------------------------------------------------------------------ #
#  GRU step (compiled)
# ------------------------------------------------------------------ #

@torch.compile
def _gru_step(x_t: Tensor, h: Tensor, W_z, W_r, W_h, U_z, U_r, U_h, W_out
              ) -> tuple[Tensor, Tensor]:
    z = torch.sigmoid(torch.bmm(W_z, x_t.unsqueeze(2)).squeeze(2)
                      + torch.bmm(U_z, h.unsqueeze(2)).squeeze(2))
    r = torch.sigmoid(torch.bmm(W_r, x_t.unsqueeze(2)).squeeze(2)
                      + torch.bmm(U_r, h.unsqueeze(2)).squeeze(2))
    h_tilde = torch.tanh(torch.bmm(W_h, x_t.unsqueeze(2)).squeeze(2)
                         + torch.bmm(U_h, (r * h).unsqueeze(2)).squeeze(2))
    h_new = (1.0 - z) * h + z * h_tilde
    u = (h_new * W_out).sum(dim=1)
    return h_new, u


# ------------------------------------------------------------------ #
#  Closed-loop rollout on the native environment
# ------------------------------------------------------------------ #

@torch.no_grad()
def simulate(genes: Tensor, ref: Tensor, inputs: tuple[str, ...], scale: Tensor,
             hidden_size: int = 8, episodes: int = 1,
             init_seed: int = None, plant_overrides: dict = None,
             return_features: bool = False, return_traj: bool = False) -> dict:
    """Batched closed-loop rollout: each of the B gene vectors runs `episodes` episodes.

    Args:
        genes: (B, n_genes) controller weights.
        ref: (nsteps,) cart-position set-point (on any device; used on CPU too).
        episodes: episodes per individual; the per-episode costs are averaged.
        init_seed: if given, initial states are the reproducible set
            ``make_init_states(episodes, init_seed)``, identical for every individual
            (fair comparisons). If None, initial states are random (training noise).
        plant_overrides: physics-parameter overrides (see PHYS_PARAMS).
        return_features: also return raw candidate signals (B*episodes, nsteps) for
            the scaling measurement.
        return_traj: also return x and theta trajectories.

    Returns:
        dict with ``cost`` (B,), ``blown_rate`` (B,) fraction of episodes that died
        early, plus optional ``features`` / ``x`` / ``theta`` / ``alive``.
    """
    device = genes.device
    B = genes.shape[0]
    E = episodes
    N = B * E
    nsteps = ref.shape[-1]
    n_in = len(inputs)

    W = unpack_gru_weights(torch.repeat_interleave(genes, E, dim=0), n_in, hidden_size)
    scale = scale.to(device=device, dtype=genes.dtype)
    # ref: (nsteps,) shared by all episodes, or (E, nsteps) one per episode slot.
    ref_np = ref.detach().cpu().numpy().astype(np.float64)
    if ref_np.ndim == 2:
        if ref_np.shape[0] != E:
            raise ValueError(f"per-episode ref needs shape ({E}, nsteps), got {ref_np.shape}")
        ep_of_env = np.arange(N) % E          # env i runs episode slot i % E
        ref_of_env = ref_np[ep_of_env]        # (N, nsteps) view by row
    else:
        ref_of_env = None

    env = _make_env(N, nsteps, plant_overrides)
    obs, _ = env.reset(seed=0 if init_seed is None else init_seed)
    if init_seed is not None:
        states = make_init_states(E, init_seed)                  # (E, 4)
        tiled = np.tile(states, (B, 1)).T.copy()                 # (4, B*E)
        env.unwrapped.state = tiled
        obs = tiled.T.astype(np.float32).copy()

    h = torch.zeros(N, hidden_size, device=device, dtype=genes.dtype)
    ie = np.zeros(N, dtype=np.float64)
    alive = np.ones(N, dtype=bool)

    cost = np.zeros(N, dtype=np.float64)
    x_prev = obs[:, 0].astype(np.float64)
    feats = ({f: np.zeros((N, nsteps), dtype=np.float32) for f in ALL_FEATURES}
             if return_features else None)
    x_traj = np.zeros((N, nsteps), dtype=np.float32) if return_traj else None
    th_traj = np.zeros((N, nsteps), dtype=np.float32) if return_traj else None

    for k in range(nsteps):
        x = obs[:, 0].astype(np.float64)
        r_k = ref_np[k] if ref_of_env is None else ref_of_env[:, k]
        e = r_k - x
        ie = ie + e * TAU * alive

        cur = {"e": e, "ie": ie, "x": x, "x_dot": obs[:, 1],
               "theta": obs[:, 2], "theta_dot": obs[:, 3]}
        if return_features:
            for f in ALL_FEATURES:
                feats[f][:, k] = cur[f]
        if return_traj:
            x_traj[:, k] = x
            th_traj[:, k] = obs[:, 2]

        # cost: tracking + smoothness while alive, flat penalty when dead
        step_cost = np.abs(e) + ALFA * np.abs(x - x_prev)
        cost += np.where(alive, step_cost, DEAD_PENALTY)
        x_prev = x

        # controller step (GPU), native discrete action
        x_t = torch.from_numpy(
            np.stack([cur[n] for n in inputs], axis=1).astype(np.float32)
        ).to(device) * scale
        h, u = _gru_step(x_t, h, *W)
        action = (u > 0).to(torch.int64).cpu().numpy()

        obs, _, term, trunc, _ = env.step(action)
        alive = alive & ~term & ~trunc

    env.close()

    cost_t = torch.from_numpy(cost.reshape(B, E)).to(device=device, dtype=torch.float32)
    blown = torch.from_numpy((~alive).reshape(B, E)).to(device=device)
    out = {"cost": cost_t.mean(dim=1), "blown_rate": blown.float().mean(dim=1)}
    if return_features:
        out["features"] = {f: torch.from_numpy(v) for f, v in feats.items()}
    if return_traj:
        out["x"] = torch.from_numpy(x_traj)
        out["theta"] = torch.from_numpy(th_traj)
        out["alive_mask"] = torch.from_numpy(alive.reshape(B, E))
    return out


# ------------------------------------------------------------------ #
#  Evaluators for IslandGA (deterministic and Bayesian)
#  The Bayesian pieces (gene split, closed-form KL, cost clip) come from
#  ga_torch.bayesian -- they are environment-independent.
# ------------------------------------------------------------------ #

from . import bayesian as _bayes                                    # noqa: E402


def make_evaluator(ref: Tensor, inputs: tuple[str, ...], scale: Tensor,
                   hidden_size: int, episodes: int = 2):
    """Deterministic GA evaluator: (B, n_genes) -> (B,) cost. Random initial states."""
    def evaluate(genes: Tensor) -> Tensor:
        return simulate(genes, ref, inputs, scale, hidden_size, episodes)["cost"]
    return evaluate


@torch.no_grad()
def simulate_mc(genes: Tensor, ref: Tensor, inputs: tuple[str, ...], scale: Tensor,
                hidden_size: int = 8, K: int = 8, episodes: int = 1,
                cost_clip: float = _bayes.COST_CLIP, init_seed: int = None,
                plant_overrides: dict = None,
                generator: torch.Generator = None) -> dict:
    """K-sample Monte-Carlo evaluation of Bayesian individuals [mu | sigma] on CartPole.

    Mirrors bayesian.simulate_mc: per-sample cost is clipped before averaging so a rare
    early-falling sample cannot make the expectation heavy-tailed.
    """
    mu, sigma = _bayes.split(genes)
    B, n_w = mu.shape
    eps = torch.randn(B, K, n_w, device=genes.device, dtype=genes.dtype, generator=generator)
    Ws = (mu.unsqueeze(1) + sigma.unsqueeze(1) * eps).reshape(B * K, n_w)

    out = simulate(Ws, ref, inputs, scale, hidden_size, episodes,
                   init_seed=init_seed, plant_overrides=plant_overrides)
    cost = out["cost"].clamp(max=cost_clip).reshape(B, K)
    return {"expected_cost": cost.mean(dim=1), "cost_std": cost.std(dim=1),
            "blown_rate": out["blown_rate"].reshape(B, K).mean(dim=1),
            "mu": mu, "sigma": sigma}


def make_sigma_only_evaluator(mu_fixed: Tensor, ref: Tensor, inputs: tuple[str, ...],
                              scale: Tensor, hidden_size: int, K: int,
                              lam: float, sigma_prior: float, episodes: int = 1):
    """Negative-ELBO evaluator over sigma-only genomes, mu fixed (Stage A on CartPole)."""
    mu_row = mu_fixed.reshape(1, -1)
    def evaluate(sigmas: Tensor) -> Tensor:
        genes = torch.cat([mu_row.expand(sigmas.shape[0], -1), sigmas], dim=1)
        out = simulate_mc(genes, ref, inputs, scale, hidden_size, K, episodes)
        kl = _bayes.kl_divergence(out["mu"], out["sigma"], sigma_prior)
        return out["expected_cost"] + lam * kl
    return evaluate


@torch.no_grad()
def weight_criticality(mu: Tensor, ref: Tensor, inputs: tuple[str, ...], scale: Tensor,
                       hidden_size: int = 8, delta: float = 0.05,
                       episodes: int = 4, init_seed: int = 123) -> Tensor:
    """Per-weight criticality on CartPole: mean |Delta cost| for a +-delta nudge of w_i.

    All 2*n_w+1 controllers are evaluated on the SAME `episodes` reproducible initial
    states (init_seed), so differences come from the weights, not the reset randomness.
    """
    n_w = mu.shape[0]
    eye = delta * torch.eye(n_w, device=mu.device, dtype=mu.dtype)
    batch = torch.cat([mu.unsqueeze(0), mu.unsqueeze(0) + eye, mu.unsqueeze(0) - eye], dim=0)
    cost = simulate(batch, ref, inputs, scale, hidden_size, episodes,
                    init_seed=init_seed)["cost"]
    base, cp, cm = cost[0], cost[1:n_w + 1], cost[n_w + 1:]
    return 0.5 * ((cp - base).abs() + (cm - base).abs())
