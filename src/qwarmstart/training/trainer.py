"""Transformer Training Loop via Gradient Updates.

Trains ParameterTransformer to predict optimal VQE initial parameters.
"""

import numpy as np
from typing import Dict, Any, List
from qwarmstart.models.parameter_transformer import ParameterTransformer


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def train_transformer(
    model: ParameterTransformer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_epochs: int = 15,
    lr: float = 0.001,
    batch_size: int = 16,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train ParameterTransformer to predict optimal VQE initial parameters.

    Parameters
    ----------
    model : ParameterTransformer
    X_train : np.ndarray shape (N, d_flat)
    y_train : np.ndarray shape (N, n_params)
    n_epochs : int
    lr : float — learning rate
    batch_size : int
    verbose : bool

    Returns
    -------
    dict with 'loss_history', 'final_loss', 'epochs_run'
    """
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
                err = np.clip(preds[i] - y_batch[i], -0.5, 0.5) / float(len(y_batch))
                x_mat = x_sample.reshape(model.seq_len, model.d_token)
                X_emb = x_mat @ model.W_in + model.b_in
                attn_out = model._attention(X_emb)
                pooled = (X_emb + attn_out).mean(axis=0)

                grad_W = np.outer(pooled, err).astype(np.float32)
                grad_b = err.astype(np.float32)

                model.W_out -= lr * np.clip(grad_W, -0.1, 0.1)
                model.b_out -= lr * np.clip(grad_b, -0.1, 0.1)

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        if verbose and ((epoch + 1) % 5 == 0 or epoch == n_epochs - 1):
            print(f"    Epoch {epoch+1:2d}/{n_epochs}: MSE Loss = {avg_loss:.6f}")

    return {
        "loss_history": loss_history,
        "final_loss": loss_history[-1],
        "epochs_run": n_epochs,
    }
