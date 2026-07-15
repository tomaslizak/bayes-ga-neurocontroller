"""Batched ODE simulation with neural network controllers on GPU.

All individuals in a population are simulated in parallel as a batch dimension.
State variables have shape (B,), output arrays have shape (B, nsteps).
"""

import torch
from torch import Tensor

# ------------------------------------------------------------------ #
#  Gene unpacking helpers
# ------------------------------------------------------------------ #

def unpack_gru_weights(genes: Tensor, input_dim: int, hidden_size: int):
    """Unpack flat gene vectors into GRU weight matrices.

    Args:
        genes: (B, n_genes) tensor.

    Returns:
        Dict of weight tensors, each with batch dimension.
    """
    B = genes.shape[0]
    idx = 0

    def get(rows, cols):
        nonlocal idx
        n = rows * cols
        w = genes[:, idx:idx + n].reshape(B, rows, cols)
        idx += n
        return w

    def get1d(size):
        nonlocal idx
        w = genes[:, idx:idx + size]
        idx += size
        return w

    W_z = get(hidden_size, input_dim)
    W_r = get(hidden_size, input_dim)
    W_h = get(hidden_size, input_dim)
    U_z = get(hidden_size, hidden_size)
    U_r = get(hidden_size, hidden_size)
    U_h = get(hidden_size, hidden_size)
    W_out = get1d(hidden_size)

    return W_z, W_r, W_h, U_z, U_r, U_h, W_out

def unpack_lstm_weights(genes: Tensor, input_dim: int, hidden_size: int):
    """Unpack flat gene vectors into LSTM weight matrices.

    Args:
        genes: (B, n_genes) tensor.

    Returns:
        Dict of weight tensors, each with batch dimension.
    """
    B = genes.shape[0]
    idx = 0

    def get(rows, cols):
        nonlocal idx
        n = rows * cols
        w = genes[:, idx:idx + n].reshape(B, rows, cols)
        idx += n
        return w

    def get1d(size):
        nonlocal idx
        w = genes[:, idx:idx + size]
        idx += size
        return w
    
    W_i = get(hidden_size, input_dim)
    W_f = get(hidden_size, input_dim)
    W_c = get(hidden_size, input_dim)
    W_o = get(hidden_size, input_dim)
    U_i = get(hidden_size, hidden_size)
    U_f = get(hidden_size, hidden_size)
    U_c = get(hidden_size, hidden_size)
    U_o = get(hidden_size, hidden_size)
    W_out = get1d(hidden_size)

    return W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, W_out

def unpack_mlp_weights(genes: Tensor, input_dim: int, hidden_size1: int, hidden_size2: int, bias: bool):
    """Unpack flat gene vectors into MLP weight matrices.

    Args:
        genes: (B, n_genes) tensor.

    Returns:
        Dict of weight tensors, each with batch dimension.
    """
    B = genes.shape[0]
    idx = 0

    def get(rows, cols):
        nonlocal idx
        n = rows * cols
        w = genes[:, idx:idx + n].reshape(B, rows, cols)
        idx += n
        return w

    def get1d(size):
        nonlocal idx
        w = genes[:, idx:idx + size]
        idx += size
        return w
    
    W1 = get(hidden_size1, input_dim)
    W2 = get(hidden_size2, hidden_size1)
    W3 = get1d(hidden_size2)

    if bias:
        B1 = get1d(hidden_size1)
        B2 = get1d(hidden_size2)
        return W1, W2, W3, B1, B2
    
    return W1, W2, W3

def unpack_mlp_bayesian_weights(genes: Tensor, input_dim: int, hidden_size1: int, hidden_size2: int):
    """Unpack flat gene vectors into MLP with bayesian weight matrices.

    Args:
        genes: (B, n_genes) tensor.

    Returns:
        Dict of weight tensors, each with batch dimension.
    """
    B = genes.shape[0]
    idx = 0

    def get(rows, cols):
        nonlocal idx
        n = rows * cols
        w = genes[:, idx:idx + n].reshape(B, rows, cols)
        idx += n
        return w

    def get1d(size):
        nonlocal idx
        w = genes[:, idx:idx + size]
        idx += size
        return w
    
    mu_1 = get(hidden_size1, input_dim)
    rho_1 = get(hidden_size1, input_dim)
    mu_2 = get(hidden_size2, hidden_size1)
    rho_2 = get(hidden_size2, hidden_size1)
    mu_3 = get1d(hidden_size2)
    rho_3 = get1d(hidden_size2)

    return mu_1, rho_1, mu_2, rho_2, mu_3, rho_3


def unpack_transformer_weights(genes: Tensor, input_dim: int, d_model: int, d_ff: int):
    """Unpack flat gene vectors into Transformer weight matrices.

    Args:
        genes: (B, n_genes) tensor.

    Returns:
        Tuple of weight tensors with batch dimension.
    """
    B = genes.shape[0]
    idx = 0

    def get(rows, cols):
        nonlocal idx
        n = rows * cols
        w = genes[:, idx:idx + n].reshape(B, rows, cols)
        idx += n
        return w

    def get1d(size):
        nonlocal idx
        w = genes[:, idx:idx + size]
        idx += size
        return w

    def get_ln(d):
        nonlocal idx
        w = genes[:, idx:idx + d].unsqueeze(1)  # (B, 1, d)
        idx += d
        return w

    W_emb = get(input_dim, d_model)
    W_q = get(d_model, d_model)
    W_k = get(d_model, d_model)
    W_v = get(d_model, d_model)
    W_ff1 = get(d_model, d_ff)
    W_ff2 = get(d_ff, d_model)
    ln1_gamma = get_ln(d_model)
    ln1_beta = get_ln(d_model)
    ln2_gamma = get_ln(d_model)
    ln2_beta = get_ln(d_model)
    W_out = get1d(d_model)

    return (W_emb, W_q, W_k, W_v, W_ff1, W_ff2,
            ln1_gamma, ln1_beta, ln2_gamma, ln2_beta, W_out)


# ------------------------------------------------------------------ #
#  Batched layer norm and softmax
# ------------------------------------------------------------------ #

def layer_norm_batch(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    """Batched layer norm. x: (B, seq, d), gamma/beta: (B, 1, d)."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return gamma * (x - mean) / (std + eps) + beta

# ------------------------------------------------------------------ #
#  Batched layer norm and softmax
# ------------------------------------------------------------------ #
def setup_precision(precision):
    """Setup PyTorch precision and AMP settings."""
    # highest|high|medium|default
    # Set matmul precision
    try:
        if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
            # New API (PyTorch >= 2.9)
            torch.backends.cuda.matmul.fp32_precision = precision
            print(f"precision succesfully set to {precision}")
        else:
            # Fallback to old API for compatibility
            torch.set_float32_matmul_precision(precision)
            print(f"precision succesfully set to {precision}.")
    except Exception:
        pass  # ignore if unsupported PyTorch