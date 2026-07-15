"""Configurable reproduction strategy for the island GA.

A *plan* is an ordered list of ``(operation, weight)`` pairs. Each generation, the population
is rebuilt by giving every operation a slice of the new population proportional to its weight.
An *operation* is a callable ``op(pop, fitness, n, ctx) -> (n, n_genes)`` that produces ``n``
new individuals (e.g. "select-by-SUS then crossover").

This replaces the previously hardcoded breeding recipe in IslandGA so the breeding strategy
can be set per run -- in particular to swap the uniform-reset mutation (which washes out an
evolved per-weight sigma) for a self-adaptive multiplicative one.

Two ready-made plans:
  * ``default_plan()``       -- reproduces the original GA behaviour.
  * ``self_adaptive_plan()`` -- Evolution-Strategy style: no uniform reset, no fresh random
                                individuals, log-normal multiplicative mutation. For evolving
                                scale parameters such as per-weight sigma.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from . import operators as ops


@dataclass
class Context:
    """Shared bounds/steps the operations need."""
    lower: Tensor
    upper: Tensor
    delta: Tensor          # additive-mutation step (per gene)


# ------------------------------------------------------------------ #
#  Operation factories  (each returns op(pop, fitness, n, ctx) -> block)
# ------------------------------------------------------------------ #

def elite():
    """Elitism: copies of the best individuals (3 of the best, then 2, then 1, ...)."""
    def op(pop, fitness, n, ctx):
        counts, rem, c = [], n, 3
        while rem > 0:
            take = min(c, rem); counts.append(take); rem -= take; c = max(1, c - 1)
        return ops.selbest(pop, fitness, n_list=counts)
    return op


def select_random():
    def op(pop, fitness, n, ctx):
        return ops.selrand(pop, fitness, n)
    return op


def crossover(n_points: int = 1):
    def op(pop, fitness, n, ctx):
        return ops.crossov(ops.selsus(pop, fitness, n), n_points)
    return op


def mutate(reset_rate: float = 0.1, add_rate: float = 0.1):
    """SUS-select, then uniform reset (mutx) + additive jitter (muta). The original operator."""
    def op(pop, fitness, n, ctx):
        w = ops.selsus(pop, fitness, n)
        w = ops.mutx(w, reset_rate, ctx.lower, ctx.upper)
        return ops.muta(w, add_rate, ctx.delta, ctx.lower, ctx.upper)
    return op


def diversity_mutate(add_rate: float = 0.12):
    def op(pop, fitness, n, ctx):
        return ops.muta(ops.seldiv(pop, fitness, n), add_rate, ctx.delta, ctx.lower, ctx.upper)
    return op


def fresh():
    """Brand-new random individuals."""
    def op(pop, fitness, n, ctx):
        return ops.genrpop(n, ctx.lower, ctx.upper)
    return op


def mutate_multiplicative(rate: float = 0.3, tau: float = 0.3, select="sus"):
    """SUS/diversity-select, then self-adaptive log-normal mutation (no reset).

    The operator for evolving positive scale parameters: it refines an existing structure
    instead of randomising it.
    """
    def op(pop, fitness, n, ctx):
        w = ops.seldiv(pop, fitness, n) if select == "div" else ops.selsus(pop, fitness, n)
        return ops.muta_log(w, rate, tau, ctx.lower, ctx.upper)
    return op


# ------------------------------------------------------------------ #
#  Plans
# ------------------------------------------------------------------ #

def default_plan():
    """The original breeding recipe (elitism, random, crossover, reset-mutation, diversity, fresh)."""
    return [
        (elite(),             0.02),
        (select_random(),     0.20),
        (crossover(),         0.28),
        (mutate(),            0.27),
        (diversity_mutate(),  0.20),
        (fresh(),             0.03),
    ]


def self_adaptive_plan(tau: float = 0.3):
    """Evolution-Strategy style: preserves and refines structure (for per-weight sigma).

    No uniform reset and no fresh random individuals -- both would erase an evolved sigma.
    Mutation is log-normal multiplicative.
    """
    return [
        (elite(),                                              0.05),
        (crossover(),                                          0.25),
        (mutate_multiplicative(rate=0.3, tau=tau),             0.45),
        (mutate_multiplicative(rate=0.6, tau=tau, select="div"), 0.25),
    ]


# ------------------------------------------------------------------ #
#  One generation
# ------------------------------------------------------------------ #

def _counts(weights: list[float], pop_size: int) -> list[int]:
    """Split pop_size across weights (largest-remainder so the counts sum exactly)."""
    total = sum(weights)
    raw = [w / total * pop_size for w in weights]
    counts = [int(x) for x in raw]
    rem = pop_size - sum(counts)
    # hand the leftover slots to the largest fractional parts
    fracs = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in range(rem):
        counts[fracs[i % len(counts)]] += 1
    return counts


def step(pop: Tensor, fitness: Tensor, plan, pop_size: int, ctx: Context) -> Tensor:
    """Produce the next generation by running every operation on its share of the population."""
    counts = _counts([w for _, w in plan], pop_size)
    blocks = [op(pop, fitness, n, ctx) for (op, _), n in zip(plan, counts) if n > 0]
    out = torch.cat(blocks, dim=0)
    return out[:pop_size]
