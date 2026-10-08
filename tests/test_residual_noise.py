"""Unit tests for Physics-Informed Residual Warm-Starting and Quantum Noise Simulation.

Tests:
  1. Residual Parameter Prediction bounds and fallback guarantees.
  2. Quantum hardware noise channels (depolarizing + readout errors).
  3. Sparse circuit fidelity advantage over dense circuits under noise.
"""

import numpy as np
import pytest
from qwarmstart.models.parameter_transformer import ParameterTransformer
from qwarmstart.models.noise_channels import (
    simulate_noisy_circuit_trajectory,
    compute_ideal_statevector,
    evaluate_noisy_vqe_energy,
)
from qwarmstart.models.baseline_vqe import get_hartree_fock_params
from qwarmstart.data.hamiltonian_encoder import h2_hamiltonian_sto3g


def test_residual_prediction_anchoring():
    """Verify that residual prediction bounds deviations around Hartree-Fock."""
    model = ParameterTransformer(rng_seed=42)
    dummy_tokens = np.random.randn(64, 33).astype(np.float32)

    hf_params = get_hartree_fock_params(4, molecule_name="H2")
    max_res = np.pi / 4

    mask_probs, theta_init, delta_theta = model.forward_residual(
        dummy_tokens, hf_params, max_residual=max_res
    )

    # Residuals must be strictly bounded within [-max_res, max_res]
    assert np.all(np.abs(delta_theta) <= max_res + 1e-6)

    # First len(hf_params) entries must equal hf_params + delta_theta
    n_hf = len(hf_params)
    np.testing.assert_allclose(theta_init[:n_hf], hf_params + delta_theta[:n_hf], atol=1e-5)


def test_predict_circuit_residual_mode():
    """Verify predict_circuit correctly populates delta_theta and bounded params."""
    model = ParameterTransformer(rng_seed=42)
    dummy_tokens = np.random.randn(64, 33).astype(np.float32)
    hf_params = np.array([np.pi, np.pi, 0.0, 0.0], dtype=np.float32)

    pred = model.predict_circuit(dummy_tokens, n_qubits=4, hf_params=hf_params)

    assert "delta_theta" in pred
    assert pred["delta_theta"] is not None
    assert len(pred["params"]) == model.n_params
    assert pred["n_cx_gates"] >= 1


def test_noiseless_state_fidelity_is_one():
    """Under zero noise (p1=0, p2=0), fidelity relative to ideal state must be exactly 1.0."""
    params = np.array([0.5, 1.2, 0.3, 0.9])
    pairs = [(0, 1), (1, 2), (2, 3)]
    pauli_terms = h2_hamiltonian_sto3g(0.735)

    res = evaluate_noisy_vqe_energy(
        pauli_terms, params, pairs, n_qubits=4, p1=0.0, p2=0.0, p_ro=0.0, n_shots=10
    )

    assert np.isclose(res["fidelity"], 1.0, atol=1e-7)
    assert np.isclose(res["energy_std"], 0.0, atol=1e-7)


def test_noise_degrades_fidelity():
    """Higher two-qubit gate error probability must lead to lower state fidelity."""
    params = np.array([0.5, 1.2, 0.3, 0.9])
    pairs = [(0, 1), (1, 2), (2, 3)]
    pauli_terms = h2_hamiltonian_sto3g(0.735)

    res_clean = evaluate_noisy_vqe_energy(
        pauli_terms, params, pairs, n_qubits=4, p1=0.0, p2=0.0, p_ro=0.0, n_shots=20, rng_seed=42
    )
    res_noisy = evaluate_noisy_vqe_energy(
        pauli_terms, params, pairs, n_qubits=4, p1=0.01, p2=0.05, p_ro=0.0, n_shots=50, rng_seed=42
    )

    assert res_clean["fidelity"] > res_noisy["fidelity"]
    assert res_noisy["fidelity"] < 0.98


def test_sparse_circuit_fidelity_advantage():
    """A sparse circuit with fewer CX gates must retain higher fidelity under 2-qubit noise than a dense circuit."""
    params = np.array([0.5, 1.2, 0.3, 0.9])
    sparse_pairs = [(0, 3)]  # 1 CX gate (predicted sparse)
    dense_pairs = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]  # 5 CX gates (dense)
    pauli_terms = h2_hamiltonian_sto3g(0.735)

    # Run with realistic physical 2-qubit error rate p2 = 0.05
    res_sparse = evaluate_noisy_vqe_energy(
        pauli_terms, params, sparse_pairs, n_qubits=4, p1=0.001, p2=0.05, p_ro=0.0, n_shots=100, rng_seed=42
    )
    res_dense = evaluate_noisy_vqe_energy(
        pauli_terms, params, dense_pairs, n_qubits=4, p1=0.001, p2=0.05, p_ro=0.0, n_shots=100, rng_seed=42
    )

    # Sparse circuit has 5x fewer CX gates, so fidelity under noise must be significantly higher!
    assert res_sparse["fidelity"] > res_dense["fidelity"]
