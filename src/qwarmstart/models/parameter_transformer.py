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
from typing import Tuple, List, Dict, Any


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


def sigmoid(x: np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def get_candidate_pairs(n_max_qubits: int = 8) -> List[Tuple[int, int]]:
    """Return canonical list of candidate qubit pairs for 2-qubit gates."""
    return [(i, j) for i in range(n_max_qubits) for j in range(i + 1, n_max_qubits)]


class ParameterTransformer:
    """Transformer for joint prediction of VQE circuit architecture (entangling mask) and initial parameters.

    Dual-head architecture:
      - Shared Transformer Encoder over Hamiltonian token sequence
      - Architecture Head: pooled representation → binary/probability mask over candidate 2-qubit pairs
      - Parameter Head: [pooled representation || predicted mask] → initial rotation angles θ

    Parameters
    ----------
    d_token : int — input token dimension (n_qubits*4 + 1)
    d_model : int — transformer hidden dimension (default 32)
    n_heads : int — attention heads (default 2)
    n_params : int — VQE parameter output dimension (default 16 for 2-layer single-qubit rotations on 8 qubits)
    seq_len : int — token sequence length (max_terms)
    n_max_qubits : int — maximum supported qubit count (default 8)
    """

    def __init__(
        self,
        d_token: int = 33,
        d_model: int = 32,
        n_heads: int = 2,
        n_params: int = 16,
        seq_len: int = 64,
        n_max_qubits: int = 8,
        rng_seed: int = 42,
    ) -> None:
        self.d_token = d_token
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.n_params = n_params
        self.seq_len = seq_len
        self.n_max_qubits = n_max_qubits
        self.candidate_pairs = get_candidate_pairs(n_max_qubits)
        self.n_candidate_pairs = len(self.candidate_pairs)

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

        # Head 1: Architecture (Mask) Head: d_model → n_candidate_pairs
        self.W_mask = rng.normal(0, 0.02, (d_model, self.n_candidate_pairs)).astype(np.float32)
        self.b_mask = np.zeros(self.n_candidate_pairs, dtype=np.float32)

        # Head 2: Parameter Head conditioned on structure: (d_model + n_candidate_pairs) → n_params
        cond_dim = d_model + self.n_candidate_pairs
        self.W_param = rng.normal(0, 0.02, (cond_dim, n_params)).astype(np.float32)
        self.b_param = np.zeros(n_params, dtype=np.float32)

        # Legacy aliases for backward compatibility with 1-head tests
        self.W_out = self.W_param[:d_model, :min(n_params, d_model)]
        self.b_out = self.b_param[:min(n_params, d_model)]

    def _attention(self, X: np.ndarray) -> np.ndarray:
        """Multi-head self-attention: X → X (same shape)."""
        head_outputs = []
        for h in range(self.n_heads):
            Q = X @ self.W_Q[h]
            K = X @ self.W_K[h]
            V = X @ self.W_V[h]

            scores = Q @ K.T / np.sqrt(self.d_k)
            attn = softmax(scores, axis=-1)
            head_out = attn @ V
            head_outputs.append(head_out)

        concat = np.concatenate(head_outputs, axis=-1)
        return concat @ self.W_O

    def _encode(self, token_matrix: np.ndarray) -> np.ndarray:
        """Run shared transformer encoder and return pooled embedding."""
        if token_matrix.ndim == 1:
            token_matrix = token_matrix.reshape(self.seq_len, self.d_token)

        X = token_matrix @ self.W_in + self.b_in
        attn_out = self._attention(X)
        X = layer_norm(X + attn_out)

        ff_out = relu(X @ self.W_ff1 + self.b_ff1) @ self.W_ff2 + self.b_ff2
        X = layer_norm(X + ff_out)

        return X.mean(axis=0)

    def forward(self, token_matrix: np.ndarray) -> np.ndarray:
        """Backward-compatible forward pass returning parameter vector."""
        mask_probs, params = self.forward_joint(token_matrix)
        return params

    def forward_joint(self, token_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Joint forward pass returning architecture mask probabilities and conditioned parameters.

        Parameters
        ----------
        token_matrix : np.ndarray shape (seq_len, d_token) or (seq_len * d_token,)

        Returns
        -------
        mask_probs : np.ndarray shape (n_candidate_pairs,) in range [0, 1]
        params : np.ndarray shape (n_params,)
        """
        pooled = self._encode(token_matrix)  # shape (d_model,)

        # Predict architecture mask probabilities
        mask_logits = pooled @ self.W_mask + self.b_mask
        mask_probs = sigmoid(mask_logits)

        # Condition parameter head on structure
        cond_vec = np.concatenate([pooled, mask_probs], axis=-1)
        params = cond_vec @ self.W_param + self.b_param

        return mask_probs, params

    def predict_circuit(
        self,
        token_matrix: np.ndarray,
        n_qubits: int,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Predict sparse entangling pairs and initial parameter values for a specific molecule.

        Parameters
        ----------
        token_matrix : np.ndarray
        n_qubits : int
        threshold : float — probability cutoff for retaining 2-qubit CNOT gate

        Returns
        -------
        dict with:
          - 'mask_probs': array of pair probabilities
          - 'selected_pairs': list of (i, j) qubit pairs with prob >= threshold within n_qubits
          - 'params': parameter array
          - 'n_cx_gates': count of predicted 2-qubit gates
          - 'n_qubits': int
        """
        mask_probs, params = self.forward_joint(token_matrix)
        selected_pairs = []
        for idx, (i, j) in enumerate(self.candidate_pairs):
            if i < n_qubits and j < n_qubits and mask_probs[idx] >= threshold:
                selected_pairs.append((i, j))

        # Fallback guarantee: if threshold removes all gates on a multi-qubit molecule,
        # retain top-1 highest scoring pair to prevent complete disconnection
        if len(selected_pairs) == 0 and n_qubits >= 2:
            valid_indices = [
                idx for idx, (i, j) in enumerate(self.candidate_pairs)
                if i < n_qubits and j < n_qubits
            ]
            if valid_indices:
                best_idx = max(valid_indices, key=lambda idx: mask_probs[idx])
                selected_pairs.append(self.candidate_pairs[best_idx])

        return {
            "mask_probs": mask_probs,
            "selected_pairs": selected_pairs,
            "params": params,
            "n_cx_gates": len(selected_pairs),
            "n_qubits": n_qubits,
        }

    def n_parameters(self) -> int:
        """Count total trainable parameters across encoder and both prediction heads."""
        total = 0
        for attr in [
            "W_in", "b_in", "W_Q", "W_K", "W_V", "W_O", "W_ff1", "b_ff1", "W_ff2", "b_ff2",
            "W_mask", "b_mask", "W_param", "b_param"
        ]:
            total += getattr(self, attr).size
        return total
