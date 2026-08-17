"""Transformer Training Loop via Adam-like Gradient Descent.

Trains ParameterTransformer to minimize MSE loss between predicted
and actual optimal VQE parameter vectors:

    L(W) = (1/N) Σ_i ||θ_pred(H_i; W) - θ_opt_i||²

Uses finite-difference gradient estimation (since we have no autograd).
This is computationally expensive but avoids PyTorch/JAX dependencies.

For real use: Replace finite-diff with PyTorch autograd for speed.
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
    n_epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 16,
    eps_fd: float = 1e-4,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train ParameterTransformer using mini-batch stochastic finite-difference gradients.

    Parameters
    ----------
    model : ParameterTransformer
    X_train : np.ndarray shape (N, d_flat)
    y_train : np.ndarray shape (N, n_params)
    n_epochs : int
    lr : float — learning rate
    batch_size : int
    eps_fd : float — finite difference step size
    verbose : bool

    Returns
    -------
    dict with 'loss_history', 'final_loss', 'epochs_run'
    """
    rng = np.random.default_rng(0)
    N = X_train.shape[0]
    loss_history = []

    # Adam optimizer state
    weight_attrs = ["W_in", "b_in", "W_Q", "W_K", "W_V", "W_O", "W_ff1", "b_ff1", "W_ff2", "b_ff2", "W_out", "b_out"]
    adam_m = {a: np.zeros_like(getattr(model, a)) for a in weight_attrs}
    adam_v = {a: np.zeros_like(getattr(model, a)) for a in weight_attrs}
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
    t = 0

    for epoch in range(n_epochs):
        idx = rng.permutation(N)
        epoch_loss = 0.0
        n_batches = 0

        for b_start in range(0, N, batch_size):
            batch_idx = idx[b_start: b_start + batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            t += 1

            # Compute batch loss
            preds = np.array([model.forward(x) for x in X_batch])
            loss = mse_loss(preds, y_batch)
            epoch_loss += loss
            n_batches += 1

            # Finite-difference gradient per weight matrix
            for attr in weight_attrs:
                W = getattr(model, attr)
                grad = np.zeros_like(W)
                flat_W = W.ravel()
                flat_grad = np.zeros_like(flat_W)

                # Sample a subset of parameters for FD (too expensive to do all)
                n_fd = min(len(flat_W), 20)
                sampled = rng.choice(len(flat_W), n_fd, replace=False)

                for i in sampled:
                    W_plus = flat_W.copy(); W_plus[i] += eps_fd
                    W_minus = flat_W.copy(); W_minus[i] -= eps_fd
                    setattr(model, attr, W_plus.reshape(W.shape))
                    preds_plus = np.array([model.forward(x) for x in X_batch])
                    setattr(model, attr, W_minus.reshape(W.shape))
                    preds_minus = np.array([model.forward(x) for x in X_batch])
                    setattr(model, attr, W)

                    flat_grad[i] = (mse_loss(preds_plus, y_batch) - mse_loss(preds_minus, y_batch)) / (2 * eps_fd)

                grad = flat_grad.reshape(W.shape)

                # Adam update
                adam_m[attr] = beta1 * adam_m[attr] + (1 - beta1) * grad
                adam_v[attr] = beta2 * adam_v[attr] + (1 - beta2) * grad ** 2
                m_hat = adam_m[attr] / (1 - beta1 ** t)
                v_hat = adam_v[attr] / (1 - beta2 ** t)
                setattr(model, attr, W - lr * m_hat / (np.sqrt(v_hat) + adam_eps))

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
            print(f"    Epoch {epoch+1:3d}/{n_epochs}: MSE Loss = {avg_loss:.6f}")

    return {
        "loss_history": loss_history,
        "final_loss": loss_history[-1],
        "epochs_run": n_epochs,
    }
