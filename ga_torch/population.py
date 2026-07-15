"""Population management and nn.Module <-> gene vector conversion."""

import torch
from torch import Tensor
import torch.nn as nn


def flatten_params(model: nn.Module) -> Tensor:
    """Flatten all model parameters into a single 1D tensor."""
    return torch.cat([p.data.reshape(-1) for p in model.parameters()])


def unflatten_params(model: nn.Module, genes: Tensor):
    """Load a flat gene vector back into model parameters (in-place)."""
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(genes[idx:idx + n].reshape(p.shape))
        idx += n


def count_params(model: nn.Module) -> int:
    """Count total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def get_space(n_genes: int, lower: float = -1.0, upper: float = 1.0,
              device: torch.device = None, dtype: torch.dtype = torch.float32) -> Tensor:
    """Create a (2, n_genes) bounds tensor.

    Args:
        n_genes: Number of genes.
        lower: Default lower bound.
        upper: Default upper bound.
        device: Target device.
        dtype: Tensor dtype.

    Returns:
        (2, n_genes) tensor where [0] = lower bounds, [1] = upper bounds.
    """
    space = torch.zeros(2, n_genes, device=device, dtype=dtype)
    space[0] = lower
    space[1] = upper
    return space


def get_space_from_model(model: nn.Module, lower: float = -1.0, upper: float = 1.0,
                         device: torch.device = None,
                         dtype: torch.dtype = torch.float32) -> Tensor:
    """Create bounds tensor sized to a model's parameter count."""
    return get_space(count_params(model), lower, upper, device, dtype)
