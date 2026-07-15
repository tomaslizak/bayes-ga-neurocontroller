"""Clean GRU neurocontroller for the 4th-order nonlinear plant.

This is the deterministic foundation for the project: one plant, one GRU controller,
one closed-loop simulator. It is deliberately small, explicit, and monitorable. The
Bayesian layer (later) builds on top of this without changing it.

Plant (Euler-integrated, 4th order, one nonlinear term):

    a4 y'''' + a3 (1 + 0.3|y|) y''' + a2 y'' + a1 y' + a0 y = b0 u + b1 u'

The controller is a small GRU whose weights are a flat gene vector (optimised by the GA).
At every step it can be fed any subset of these six candidate signals:

    e    tracking error            r - y
    ie   integral of error         sum e * Ts
    y    plant output
    d1y  output 1st derivative      (y[k]-y[k-1])/Ts
    d2y  output 2nd derivative      (d1y[k]-d1y[k-1])/Ts
    du   control 1st derivative     (u[k]-u[k-1])/Ts

Each fed signal is multiplied by a per-signal SCALE so the GRU (whose weights live in
[-1, 1]) sees inputs of order ~1. Choosing those scales correctly is the point of the
companion measurement step -- `simulate(..., return_features=True)` returns every raw
(unscaled) signal so its real range can be measured.
"""

import torch
from torch import Tensor

from .simulation import unpack_gru_weights

# ------------------------------------------------------------------ #
#  Plant + problem definition
# ------------------------------------------------------------------ #

PLANT = dict(a0=1.0, a1=3.0, a2=8.0, a3=1.0, a4=1.0, b0=1.0, b1=0.1,
             nl_gain=0.3, u_max=1000.0)

# The six candidate controller inputs, in a fixed canonical order.
ALL_FEATURES = ("e", "ie", "y", "d1y", "d2y", "du")

# The chosen controller input set (decided by the head-to-head in train_baseline.py):
# error, integral, output, velocity, acceleration. `du` was dropped -- pathologically
# spiky (max ~13600) and it did not help. This is the foundation for the Bayesian layer.
INPUTS = ("e", "ie", "y", "d1y", "d2y")

# Per-signal input scales, set by the 1/p99 rule (each signal divided by the 99th percentile
# of |signal| measured from a trained, well-tracking controller -> inputs sit in [-1,1] ~99%
# of the time). Measured by train_baseline.py; see results/baseline_measure.json.
DEFAULT_SCALE = {
    "e":   1/86.4,    # error
    "ie":  1/175.1,   # integral of error
    "y":   1/100.0,   # output
    "d1y": 1/43.5,    # output 1st derivative
    "d2y": 1/40.4,    # output 2nd derivative
    "du":  1/2130.1,  # control 1st derivative
}

Ts_DEFAULT = 0.05


def get_reference(nsteps: int = 4000, Ts: float = Ts_DEFAULT,
                  device: torch.device = None) -> Tensor:
    """Piecewise-constant setpoint: 0 -> 50 -> 0 -> 100 -> 0 (seconds 1/5/45/85/140)."""
    t = torch.arange(nsteps, device=device, dtype=torch.float32) * Ts
    w = torch.zeros(nsteps, device=device, dtype=torch.float32)
    w[t >= 5] = 50.0
    w[t >= 45] = 0.0
    w[t >= 85] = 100.0
    w[t >= 140] = 0.0
    return w


def gene_count(n_inputs: int, hidden_size: int) -> int:
    """Flat gene-vector length of a GRU with `n_inputs` inputs and `hidden_size` units."""
    return 3 * hidden_size * n_inputs + 3 * hidden_size * hidden_size + hidden_size


# ------------------------------------------------------------------ #
#  GRU cell (compiled)
# ------------------------------------------------------------------ #

@torch.compile
def _gru_step(x_t: Tensor, h: Tensor, W_z, W_r, W_h, U_z, U_r, U_h, W_out,
              u_max: float) -> tuple[Tensor, Tensor]:
    """One batched GRU step -> (new hidden state, bounded scalar control)."""
    z = torch.sigmoid(torch.bmm(W_z, x_t.unsqueeze(2)).squeeze(2)
                      + torch.bmm(U_z, h.unsqueeze(2)).squeeze(2))
    r = torch.sigmoid(torch.bmm(W_r, x_t.unsqueeze(2)).squeeze(2)
                      + torch.bmm(U_r, h.unsqueeze(2)).squeeze(2))
    h_tilde = torch.tanh(torch.bmm(W_h, x_t.unsqueeze(2)).squeeze(2)
                         + torch.bmm(U_h, (r * h).unsqueeze(2)).squeeze(2))
    h_new = (1.0 - z) * h + z * h_tilde
    u = (h_new * W_out).sum(dim=1)
    return h_new, torch.tanh(u) * u_max


# ------------------------------------------------------------------ #
#  Closed-loop simulation
# ------------------------------------------------------------------ #

@torch.no_grad()
def simulate(genes: Tensor, ref: Tensor,
             inputs: tuple[str, ...], scale: Tensor,
             hidden_size: int = 8, Ts: float = Ts_DEFAULT,
             return_features: bool = False,
             plant_overrides: dict = None) -> dict:
    """Batched closed-loop rollout of GRU controllers on the plant.

    Args:
        genes: (B, n_genes) gene vectors, with n_genes == gene_count(len(inputs), hidden_size).
        ref: (nsteps,) setpoint on device.
        inputs: which candidate signals to feed the GRU, e.g. ("e","ie","y","d1y","d2y","du").
        scale: (len(inputs),) tensor of per-input scales aligned to `inputs`.
        return_features: if True, also return every raw (unscaled) candidate signal,
            each (B, nsteps), for range measurement.
        plant_overrides: optional dict of plant-parameter overrides (e.g. {"a1": 3.6})
            merged on top of PLANT, for robustness/perturbation tests. None = nominal plant.

    Returns:
        dict with (B, nsteps) tensors e, y, u; a (B,) bool `blown` divergence mask; and,
        if requested, `features`: {name: (B, nsteps)} of raw signals.
    """
    p = PLANT if plant_overrides is None else {**PLANT, **plant_overrides}
    a0, a1, a2, a3, a4 = p["a0"], p["a1"], p["a2"], p["a3"], p["a4"]
    b0, b1, nl, u_max = p["b0"], p["b1"], p["nl_gain"], p["u_max"]

    device, dtype = genes.device, genes.dtype
    B = genes.shape[0]
    nsteps = ref.shape[0]
    n_inputs = len(inputs)

    W_z, W_r, W_h, U_z, U_r, U_h, W_out = unpack_gru_weights(genes, n_inputs, hidden_size)

    # Plant + bookkeeping state, all (B,)
    y0 = torch.zeros(B, device=device, dtype=dtype)
    dy0 = torch.zeros_like(y0); ddy0 = torch.zeros_like(y0); dddy0 = torch.zeros_like(y0)
    u0 = torch.zeros_like(y0); u1 = torch.zeros_like(y0)
    ie = torch.zeros_like(y0); d1y_prev = torch.zeros_like(y0)
    h = torch.zeros(B, hidden_size, device=device, dtype=dtype)
    blown = torch.zeros(B, device=device, dtype=torch.bool)

    e_out = torch.zeros(B, nsteps, device=device, dtype=dtype)
    y_out = torch.zeros(B, nsteps, device=device, dtype=dtype)
    u_out = torch.zeros(B, nsteps, device=device, dtype=dtype)
    feats = ({f: torch.zeros(B, nsteps, device=device, dtype=dtype) for f in ALL_FEATURES}
             if return_features else None)

    scale = scale.to(device=device, dtype=dtype)

    for k in range(1, nsteps):
        ww = ref[k]

        # --- plant: one cascaded Euler step ---
        du0 = (u0 - u1) / Ts
        ddddy = (b0 * u0 + b1 * du0
                 - a3 * (1.0 + nl * torch.abs(y0)) * dddy0
                 - a2 * ddy0 - a1 * dy0 - a0 * y0) / a4
        dddy0 = dddy0 + ddddy * Ts
        ddy0 = ddy0 + dddy0 * Ts
        dy0 = dy0 + ddy0 * Ts
        y_new = y0 + dy0 * Ts

        y_out[:, k] = y_new
        u_out[:, k] = u0

        u1 = u0
        y_prev = y0
        y0 = y_new

        e_k = ww - y0
        e_out[:, k] = e_k
        ie = ie + e_k * Ts
        d1y = (y0 - y_prev) / Ts
        d2y = (d1y - d1y_prev) / Ts
        d1y_prev = d1y

        blown = blown | (torch.abs(y0) > 1e6)

        current = {"e": e_k, "ie": ie, "y": y0, "d1y": d1y, "d2y": d2y, "du": du0}
        if return_features:
            for f in ALL_FEATURES:
                feats[f][:, k] = current[f]

        # build scaled controller input (B, n_inputs)
        x_t = torch.stack([current[name] for name in inputs], dim=1) * scale
        h, u0 = _gru_step(x_t, h, W_z, W_r, W_h, U_z, U_r, U_h, W_out, u_max)

    return {"e": e_out, "y": y_out, "u": u_out, "blown": blown, "features": feats}


# ------------------------------------------------------------------ #
#  Fitness
# ------------------------------------------------------------------ #

def fitness(sim: dict, alfa: float = 30.0, penalty: float = 1e8) -> Tensor:
    """Cost = sum|e| + alfa*sum|diff(y)|, with `penalty` for diverged / non-finite runs.

    Lower is better (the GA minimises). `sim` is the dict returned by `simulate`.
    """
    e, y = sim["e"], sim["y"]
    cost = torch.sum(torch.abs(e), dim=1) + alfa * torch.sum(torch.abs(torch.diff(y, dim=1)), dim=1)
    bad = sim["blown"] | torch.isnan(cost) | torch.isinf(cost)
    return torch.where(bad, torch.full_like(cost, penalty), cost)


def scale_tensor(inputs: tuple[str, ...], scale_map: dict = None,
                 device: torch.device = None, dtype: torch.dtype = torch.float32) -> Tensor:
    """Build the (len(inputs),) scale tensor for `inputs` from a {name: factor} map."""
    scale_map = scale_map or DEFAULT_SCALE
    return torch.tensor([scale_map[name] for name in inputs], device=device, dtype=dtype)
