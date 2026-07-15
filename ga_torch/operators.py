"""Vectorized genetic algorithm operators using PyTorch tensors.

All functions operate on tensors and stay on device (no CPU transfers).
Population tensors have shape (pop_size, n_genes).
Fitness tensors have shape (pop_size,).
"""

import torch
from torch import Tensor


def genrpop(n: int, lower: Tensor, upper: Tensor) -> Tensor:
    """Generate random population within bounds.

    Args:
        n: Number of individuals.
        lower: (n_genes,) lower bounds.
        upper: (n_genes,) upper bounds.

    Returns:
        (n, n_genes) population tensor.
    """
    return torch.rand(n, lower.shape[0], device=lower.device, dtype=lower.dtype) * (upper - lower) + lower


def selbest(pop: Tensor, fitness: Tensor, n_list: list[int]) -> Tensor:
    """Elitist selection. Returns copies of top individuals according to n_list.

    n_list[0] copies of rank-0, n_list[1] copies of rank-1, etc.
    """
    sorted_idx = torch.argsort(fitness)
    indices = []
    for rank, copies in enumerate(n_list):
        indices.extend([sorted_idx[rank]] * copies)
    return pop[torch.stack(indices)].clone()


def selrand(pop: Tensor, fitness: Tensor, n: int) -> Tensor:
    """Random selection of n individuals (with replacement)."""
    indices = torch.randint(0, pop.shape[0], (n,), device=pop.device)
    return pop[indices].clone()


def selsus(pop: Tensor, fitness: Tensor, n: int) -> Tensor:
    """Stochastic Universal Sampling selection.

    Weights are inversely proportional to fitness (minimization).
    """
    # Compute weights: lower fitness = higher weight
    shifted = fitness - fitness.min() + 1.0
    weights = 1.0 / shifted
    weights = weights / weights.sum()

    # Cumulative distribution
    cumsum = torch.cumsum(weights, dim=0)

    # Evenly-spaced pointers with random offset
    offset = torch.rand(1, device=pop.device, dtype=pop.dtype) / n
    pointers = offset + torch.arange(n, device=pop.device, dtype=pop.dtype) / n

    # Find which individual each pointer selects
    indices = torch.searchsorted(cumsum, pointers)
    indices = indices.clamp(max=pop.shape[0] - 1)

    return pop[indices].clone()


def seldiv(pop: Tensor, fitness: Tensor, n: int) -> Tensor:
    """Diversity-preserving selection.

    Selects n individuals that are most different from the best individual (L1 distance).
    """
    best_idx = torch.argmin(fitness)
    best = pop[best_idx]

    # L1 distance from best
    distances = torch.sum(torch.abs(pop - best), dim=1)

    # Select top-n most diverse
    _, top_indices = torch.topk(distances, min(n, pop.shape[0]))

    selected = pop[top_indices].clone()
    # If n > available, repeat to fill
    if selected.shape[0] < n:
        repeats = (n + selected.shape[0] - 1) // selected.shape[0]
        selected = selected.repeat(repeats, 1)[:n]
    return selected


def crossov(pop: Tensor, n_points: int = 1) -> Tensor:
    """N-point crossover on shuffled pairs.

    Pairs are formed by shuffling, then segments are swapped.
    Population size should be even.
    """
    n_ind, n_genes = pop.shape
    n_pairs = n_ind // 2

    # Shuffle pairing
    perm = torch.randperm(n_ind, device=pop.device)
    pop = pop[perm]

    parent1 = pop[:n_pairs]  # (n_pairs, n_genes)
    parent2 = pop[n_pairs:2 * n_pairs]

    # Generate crossover points for each pair
    # For 1-point: one random cut point per pair
    points = torch.sort(
        torch.randint(1, n_genes, (n_pairs, n_points), device=pop.device),
        dim=1
    )[0]  # (n_pairs, n_points)

    # Build swap mask: for each pair, alternate segments
    gene_idx = torch.arange(n_genes, device=pop.device).unsqueeze(0)  # (1, n_genes)
    # Count how many crossover points each gene position is past
    # shape: (n_pairs, n_genes)
    counts = (gene_idx >= points.unsqueeze(2)).sum(dim=1)
    swap_mask = (counts % 2 == 1)  # swap on odd segments

    # Perform crossover
    child1 = torch.where(swap_mask, parent2, parent1)
    child2 = torch.where(swap_mask, parent1, parent2)

    result = pop.clone()
    result[:n_pairs] = child1
    result[n_pairs:2 * n_pairs] = child2
    return result


def mutx(pop: Tensor, rate: float, lower: Tensor, upper: Tensor) -> Tensor:
    """Global (reset) mutation. Selected genes are replaced with random values.

    Args:
        pop: (pop_size, n_genes) population.
        rate: Probability of mutating each gene.
        lower: (n_genes,) lower bounds.
        upper: (n_genes,) upper bounds.

    Returns:
        Mutated population (new tensor).
    """
    mask = torch.rand_like(pop) < rate
    random_vals = torch.rand_like(pop) * (upper - lower) + lower
    return torch.where(mask, random_vals, pop)


def muta(pop: Tensor, rate: float, delta: Tensor,
         lower: Tensor, upper: Tensor) -> Tensor:
    """Additive mutation. Selected genes are perturbed by ±delta.

    Args:
        pop: (pop_size, n_genes) population.
        rate: Probability of mutating each gene.
        delta: (n_genes,) max perturbation per gene.
        lower: (n_genes,) lower bounds.
        upper: (n_genes,) upper bounds.

    Returns:
        Mutated population (new tensor).
    """
    mask = torch.rand_like(pop) < rate
    perturbation = (torch.rand_like(pop) * 2 - 1) * delta
    result = pop + mask.float() * perturbation
    return torch.clamp(result, lower, upper)


def muta_log(pop: Tensor, rate: float, tau: float,
             lower: Tensor, upper: Tensor) -> Tensor:
    """Self-adaptive (log-normal) multiplicative mutation: gene *= exp(tau * N(0,1)).

    Unlike ``mutx``/``muta`` this neither resets nor adds a fixed step -- it scales the gene,
    so an existing *structure* across genes is preserved and only refined. This is the
    Evolution-Strategy style operator for evolving strictly-positive scale parameters
    (e.g. per-weight sigma), where uniform reset would wash the structure out.

    Args:
        pop: (pop_size, n_genes) population (genes assumed positive).
        rate: Probability of mutating each gene.
        tau: Log-normal step size (std of the exponent).
        lower, upper: (n_genes,) bounds for clamping.
    """
    mask = torch.rand_like(pop) < rate
    factor = torch.exp(tau * torch.randn_like(pop))
    result = torch.where(mask, pop * factor, pop)
    return torch.clamp(result, lower, upper)
