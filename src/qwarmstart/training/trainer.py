"""Transformer Training Loop via Gradient Updates.

Trains ParameterTransformer to predict optimal VQE initial parameters.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from qwarmstart.models.parameter_transformer import ParameterTransformer


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def binary_cross_entropy(p_pred: np.ndarray, p_true: np.ndarray, eps: float = 1e-7) -> float:
    p_pred = np.clip(p_pred, eps, 1.0 - eps)
    return float(-np.mean(p_true * np.log(p_pred) + (1.0 - p_true) * np.log(1.0 - p_pred)))


def compute_connectivity_penalty(
    mask: np.ndarray,
    candidate_pairs: List[Tuple[int, int]],
    n_qubits: int,
) -> Tuple[float, np.ndarray]:
    """Compute soft penalty ensuring each active qubit has non-zero connectivity."""
    grad = np.zeros_like(mask)
    degree = np.zeros(n_qubits, dtype=np.float32)

    for idx, (i, j) in enumerate(candidate_pairs):
        if i < n_qubits and j < n_qubits:
            degree[i] += mask[idx]
            degree[j] += mask[idx]

    penalty = 0.0
    for q in range(n_qubits):
        if degree[q] < 1.0:
            diff = 1.0 - degree[q]
            penalty += float(diff ** 2)
            # Derivative w.r.t mask[idx]
            for idx, (i, j) in enumerate(candidate_pairs):
                if (i == q or j == q) and i < n_qubits and j < n_qubits:
                    grad[idx] -= 2.0 * diff

    return penalty, grad


def train_joint_transformer(
    model: ParameterTransformer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    mask_train: np.ndarray,
    n_epochs: int = 25,
    lr: float = 0.01,
    lambda_sparse: float = 0.05,
    lambda_conn: float = 0.02,
    batch_size: int = 16,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train ParameterTransformer with joint multi-objective loss for structure + parameters.

    Loss = MSE(θ_pred, θ_true) + BCE(m_pred, m_true) + λ_sparse * mean(m_pred) + λ_conn * Connectivity(m_pred)
    """
    rng = np.random.default_rng(42)
    N = X_train.shape[0]
    param_loss_history = []
    mask_loss_history = []
    total_loss_history = []

    for epoch in range(n_epochs):
        idx = rng.permutation(N)
        epoch_param_loss = 0.0
        epoch_mask_loss = 0.0
        n_batches = 0

        for b_start in range(0, N, batch_size):
            batch_idx = idx[b_start: b_start + batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            m_batch = mask_train[batch_idx]
            n_batches += 1

            for i, x_sample in enumerate(X_batch):
                y_true = y_batch[i]
                m_true = m_batch[i]

                # Forward pass
                pooled = model._encode(x_sample)
                mask_logits = pooled @ model.W_mask + model.b_mask
                m_pred = 1.0 / (1.0 + np.exp(-np.clip(mask_logits, -50.0, 50.0)))

                cond_vec = np.concatenate([pooled, m_pred], axis=-1)
                p_pred = cond_vec @ model.W_param + model.b_param

                # Losses
                p_err = (p_pred - y_true)
                param_loss = float(np.mean(p_err ** 2))

                bce = binary_cross_entropy(m_pred, m_true)
                conn_pen, conn_grad = compute_connectivity_penalty(m_pred, model.candidate_pairs, model.n_max_qubits)
                mask_loss = bce + lambda_sparse * float(np.mean(m_pred)) + lambda_conn * conn_pen

                epoch_param_loss += param_loss
                epoch_mask_loss += mask_loss

                # Gradients for Parameter Head
                norm_p_err = np.clip(p_err / float(len(y_true)), -0.5, 0.5)
                grad_W_param = np.outer(cond_vec, norm_p_err).astype(np.float32)
                grad_b_param = norm_p_err.astype(np.float32)

                # Gradients for Mask Head
                bce_grad = (m_pred - m_true) / float(len(m_true))
                sparse_grad = lambda_sparse / float(len(m_true))
                mask_delta = (bce_grad + sparse_grad + lambda_conn * conn_grad) * m_pred * (1.0 - m_pred)
                mask_delta = np.clip(mask_delta, -0.5, 0.5)

                grad_W_mask = np.outer(pooled, mask_delta).astype(np.float32)
                grad_b_mask = mask_delta.astype(np.float32)

                # Parameter updates
                model.W_param -= lr * np.clip(grad_W_param, -0.1, 0.1)
                model.b_param -= lr * np.clip(grad_b_param, -0.1, 0.1)
                model.W_mask -= lr * np.clip(grad_W_mask, -0.1, 0.1)
                model.b_mask -= lr * np.clip(grad_b_mask, -0.1, 0.1)

                # Synchronize legacy aliases
                model.W_out = model.W_param[:model.d_model, :min(model.n_params, model.d_model)]
                model.b_out = model.b_param[:min(model.n_params, model.d_model)]

        avg_param_l = epoch_param_loss / max(n_batches, 1)
        avg_mask_l = epoch_mask_loss / max(n_batches, 1)
        param_loss_history.append(avg_param_l)
        mask_loss_history.append(avg_mask_l)
        total_loss_history.append(avg_param_l + avg_mask_l)

        if verbose and ((epoch + 1) % 5 == 0 or epoch == n_epochs - 1):
            print(f"    Epoch {epoch+1:2d}/{n_epochs}: Param Loss = {avg_param_l:.5f} | Mask Loss = {avg_mask_l:.5f} | Total = {avg_param_l + avg_mask_l:.5f}")

    return {
        "param_loss_history": param_loss_history,
        "mask_loss_history": mask_loss_history,
        "total_loss_history": total_loss_history,
        "final_loss": total_loss_history[-1],
        "epochs_run": n_epochs,
    }


def train_transformer(
    model: ParameterTransformer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_epochs: int = 15,
    lr: float = 0.001,
    batch_size: int = 16,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train ParameterTransformer to predict optimal VQE initial parameters (backward compatible)."""
    rng = np.random.default_rng(42)
    N = X_train.shape[0]
    loss_history = []

    for epoch in range(n_epochs):
        idx = rng.permutation(N)
        epoch_loss = 0.0
        n_batches = 0

        for b_start in range(0, N, batch_size):
            batch_idx = idx[b_start: b_start + batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            n_batches += 1

            preds = np.array([model.forward(x) for x in X_batch], dtype=np.float32)
            loss = mse_loss(preds, y_batch)
            epoch_loss += loss

            for i, x_sample in enumerate(X_batch):
                err = np.clip(preds[i] - y_batch[i], -0.5, 0.5) / float(len(y_batch[i]))
                pooled = model._encode(x_sample)
                mask_probs, _ = model.forward_joint(x_sample)
                cond_vec = np.concatenate([pooled, mask_probs], axis=-1)

                grad_W = np.outer(cond_vec, err).astype(np.float32)
                grad_b = err.astype(np.float32)

                model.W_param -= lr * np.clip(grad_W, -0.1, 0.1)
                model.b_param -= lr * np.clip(grad_b, -0.1, 0.1)
                model.W_out = model.W_param[:model.d_model, :min(model.n_params, model.d_model)]
                model.b_out = model.b_param[:min(model.n_params, model.d_model)]

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        if verbose and ((epoch + 1) % 5 == 0 or epoch == n_epochs - 1):
            print(f"    Epoch {epoch+1:2d}/{n_epochs}: MSE Loss = {avg_loss:.6f}")

    return {
        "loss_history": loss_history,
        "final_loss": loss_history[-1],
        "epochs_run": n_epochs,
    }
