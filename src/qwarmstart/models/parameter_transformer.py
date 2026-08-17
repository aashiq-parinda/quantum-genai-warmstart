"""Tiny Transformer Encoder for VQE Parameter Prediction.

Architecture: Hamiltonian Token Sequence → Multi-Head Self-Attention → MLP Head → θ_pred

This is a pure NumPy transformer implementing:
  1. Linear token projection (d_token → d_model)
  2. Multi-head self-attention (h heads, d_k = d_model / h)
  3. Feed-forward MLP (d_model → 4*d_model → d_model)
  4. Global average pooling over sequence
  5. Output head (d_model → n_params)

All implemented from scratch in NumPy for:
  - Zero external dependencies
  - Full interpretability of every operation
  - Educational transparency

Parameters ~12K (extremely lightweight)

Reference architecture:
  Vaswani et al. (2017) "Attention Is All You Need". arXiv:1706.03762
"""

import numpy as np
from typing import Tuple


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_clipped = np.clip(x, -50.0, 50.0)
    x_shift = x_clipped - np.max(x_clipped, axis=axis, keepdims=True)
    e = np.exp(x_shift)
    s = np.sum(e, axis=axis, keepdims=True)
    return e / np.maximum(s, 1e-12)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True) + eps
    normalized = (x - mean) / std
    return np.clip(normalized, -10.0, 10.0)


class ParameterTransformer:
    """Tiny Transformer for predicting VQE initial parameters from Hamiltonian tokens.

    Parameters
    ----------
    d_token : int — input token dimension (n_qubits*4 + 1)
    d_model : int — transformer hidden dimension (default 32)
    n_heads : int — attention heads (default 2)
    n_params : int — VQE parameter output dimension
    seq_len : int — token sequence length (max_terms)
    """

    def __init__(
        self,
        d_token: int = 17,
        d_model: int = 32,
        n_heads: int = 2,
        n_params: int = 4,
        seq_len: int = 32,
        rng_seed: int = 42,
    ) -> None:
        self.d_token = d_token
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.n_params = n_params
        self.seq_len = seq_len

        rng = np.random.default_rng(rng_seed)
        scale = np.sqrt(2.0 / (d_token + d_model))

        # Token projection
        self.W_in = rng.normal(0, scale, (d_token, d_model)).astype(np.float32)
        self.b_in = np.zeros(d_model, dtype=np.float32)

        # Multi-head attention: Q, K, V projections
        self.W_Q = rng.normal(0, scale, (n_heads, d_model, self.d_k)).astype(np.float32)
        self.W_K = rng.normal(0, scale, (n_heads, d_model, self.d_k)).astype(np.float32)
        self.W_V = rng.normal(0, scale, (n_heads, d_model, self.d_k)).astype(np.float32)
        self.W_O = rng.normal(0, scale, (n_heads * self.d_k, d_model)).astype(np.float32)

        # Feed-forward network: d_model → 4*d_model → d_model
        ff_dim = 4 * d_model
        self.W_ff1 = rng.normal(0, scale, (d_model, ff_dim)).astype(np.float32)
        self.b_ff1 = np.zeros(ff_dim, dtype=np.float32)
        self.W_ff2 = rng.normal(0, scale, (ff_dim, d_model)).astype(np.float32)
        self.b_ff2 = np.zeros(d_model, dtype=np.float32)

        # Output head: d_model → n_params
        self.W_out = rng.normal(0, 0.01, (d_model, n_params)).astype(np.float32)
        self.b_out = np.zeros(n_params, dtype=np.float32)

    def _attention(self, X: np.ndarray) -> np.ndarray:
        """Multi-head self-attention: X → X (same shape).

        X shape: (seq_len, d_model)
        """
        head_outputs = []
        for h in range(self.n_heads):
            Q = X @ self.W_Q[h]   # (seq, d_k)
            K = X @ self.W_K[h]
            V = X @ self.W_V[h]

            scores = Q @ K.T / np.sqrt(self.d_k)   # (seq, seq)
            attn = softmax(scores, axis=-1)
            head_out = attn @ V                     # (seq, d_k)
            head_outputs.append(head_out)

        concat = np.concatenate(head_outputs, axis=-1)  # (seq, n_heads*d_k)
        return concat @ self.W_O                        # (seq, d_model)

    def forward(self, token_matrix: np.ndarray) -> np.ndarray:
        """Forward pass: token_matrix → predicted VQE parameters.

        Parameters
        ----------
        token_matrix : np.ndarray shape (seq_len, d_token) or (seq_len * d_token,)

        Returns
        -------
        np.ndarray shape (n_params,) — predicted parameter initialization
        """
        # Reshape if flattened
        if token_matrix.ndim == 1:
            token_matrix = token_matrix.reshape(self.seq_len, self.d_token)

        # Token projection: (seq, d_token) → (seq, d_model)
        X = token_matrix @ self.W_in + self.b_in

        # Transformer block with residual connections + LayerNorm
        attn_out = self._attention(X)
        X = layer_norm(X + attn_out)

        ff_out = relu(X @ self.W_ff1 + self.b_ff1) @ self.W_ff2 + self.b_ff2
        X = layer_norm(X + ff_out)

        # Global average pooling
        pooled = X.mean(axis=0)  # (d_model,)

        # Output head
        params_pred = pooled @ self.W_out + self.b_out   # (n_params,)
        return params_pred

    def n_parameters(self) -> int:
        """Count total trainable parameters."""
        total = 0
        for attr in ["W_in", "b_in", "W_Q", "W_K", "W_V", "W_O", "W_ff1", "b_ff1", "W_ff2", "b_ff2", "W_out", "b_out"]:
            total += getattr(self, attr).size
        return total
