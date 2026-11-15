"""gymnasium MountainCarContinuous-v0 dynamics, batched in torch and checked against native.

Native environment (gymnasium ContinuousMountainCarEnv): state (position x, velocity v),
continuous action a in [-1, 1], update per step

    v' = clip(v + clip(a, -1, 1) * power - 0.0025 cos(3 x), -max_speed, max_speed)
    x' = clip(x + v', min_position, max_position)
    v' = 0                      if the car sits on the left wall with v' < 0
    terminated = x' >= goal_position and v' >= goal_velocity

reward per step = -0.1 a^2, plus 100 on the terminating step; the episode is truncated at
999 steps; reset draws x ~ U(-0.6, -0.4), v = 0. Gymnasium ships no batched version, so
`plant_step` implements the same rule for a batch of cars. `validate_against_native()`
drives the native environment and `plant_step` with identical actions and reports the
largest deviation in state, the termination steps and the returns.
"""

import numpy as np
import torch
from torch import Tensor

# ------------------------------------------------------------------ #
#  Native constants (gymnasium ContinuousMountainCarEnv defaults)
# ------------------------------------------------------------------ #

PHYS = dict(power=0.0015, gravity=0.0025, max_speed=0.07,
            min_position=-1.2, max_position=0.6,
            goal_position=0.45, goal_velocity=0.0)
NSTEPS = 999                      # native TimeLimit
RESET_LOW, RESET_HIGH = -0.6, -0.4
GOAL_REWARD, ACTION_COST = 100.0, 0.1

PHYS_PARAMS = ("power", "gravity")


def plant_step(x: Tensor, v: Tensor, a: Tensor, p: dict = PHYS):
    """One native step for a batch of cars. Returns (x', v', reached)."""
    force = torch.clamp(a, -1.0, 1.0)
    v = v + force * p["power"] - p["gravity"] * torch.cos(3.0 * x)
    v = torch.clamp(v, -p["max_speed"], p["max_speed"])
    x = torch.clamp(x + v, p["min_position"], p["max_position"])
    v = torch.where((x == p["min_position"]) & (v < 0), torch.zeros_like(v), v)
    reached = (x >= p["goal_position"]) & (v >= p["goal_velocity"])
    return x, v, reached


# ------------------------------------------------------------------ #
#  Validation against the native gymnasium implementation
# ------------------------------------------------------------------ #

def validate_against_native(n_episodes: int = 20, seed: int = 0,
                            dtype: torch.dtype = torch.float32) -> dict:
    """Run native episodes and the batched port side by side with identical actions.

    Half of the episodes use uniform random actions (the car stays in the valley), half an
    energy-pumping policy computed from the native state (a = sign(v), 20% random actions),
    which reaches the goal and so exercises termination and the terminal reward.

    Returns the largest absolute state deviation over all steps, whether every episode
    terminated on the same step, and the largest difference in episode return.
    """
    import gymnasium as gym
    rng = np.random.default_rng(seed)
    env = gym.make("MountainCarContinuous-v0")
    worst, same_end, worst_ret, n_reached, n_steps = 0.0, True, 0.0, 0, 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        x = torch.tensor([float(obs[0])], dtype=dtype)
        v = torch.tensor([float(obs[1])], dtype=dtype)
        ret_native, ret_port, done_port = 0.0, 0.0, False
        for k in range(NSTEPS):
            if ep % 2 == 0 or rng.random() < 0.2:
                a = float(rng.uniform(-1.0, 1.0))
            else:
                a = 1.0 if obs[1] >= 0 else -1.0
            obs, r, term, trunc, _ = env.step(np.array([a], dtype=np.float32))
            ret_native += float(r)
            x, v, reached = plant_step(x, v, torch.tensor([a], dtype=dtype))
            ret_port += -ACTION_COST * float(np.float32(a)) ** 2 + (GOAL_REWARD if reached.item() else 0.0)
            done_port = bool(reached.item())
            worst = max(worst, abs(x.item() - float(obs[0])), abs(v.item() - float(obs[1])))
            n_steps += 1
            if term != done_port:
                same_end = False
            if term or trunc or done_port:
                break
        n_reached += int(done_port)
        worst_ret = max(worst_ret, abs(ret_native - ret_port))
    env.close()
    return dict(max_state_deviation=worst, same_termination=same_end,
                max_return_difference=worst_ret, episodes=n_episodes,
                episodes_reaching_goal=n_reached, steps=n_steps)
