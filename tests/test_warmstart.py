"""Unit tests for quantum-genai-warmstart package."""
import numpy as np
import pytest
from qwarmstart.data.hamiltonian_encoder import (
    encode_pauli_string, encode_hamiltonian, hamiltonian_to_flat_vector,
    h2_hamiltonian_sto3g, random_hamiltonian,
)
from qwarmstart.data.dataset_generator import evaluate_vqe_energy, generate_dataset
from qwarmstart.models.parameter_transformer import ParameterTransformer
from qwarmstart.models.baseline_vqe import run_baseline_vqe


class TestHamiltonianEncoder:
    def test_encode_pauli_string_shape(self):
        enc = encode_pauli_string("IXYZ", 4)
        assert enc.shape == (16,)  # 4 qubits * 4 Pauli

    def test_encode_pauli_string_one_hot(self):
        enc = encode_pauli_string("IIII", 4)
        assert enc.sum() == 4.0  # 4 qubits, each selects 'I' (index 0)

    def test_encode_hamiltonian_shape(self):
        terms = h2_hamiltonian_sto3g()
        tokens = encode_hamiltonian(terms, 4, max_terms=32)
        assert tokens.shape == (32, 17)  # (max_terms, n_qubits*4 + 1)

    def test_flat_vector_shape(self):
        terms = h2_hamiltonian_sto3g()
        v = hamiltonian_to_flat_vector(terms, 4, max_terms=32)
        assert v.shape == (32 * 17,)

    def test_invalid_pauli_char(self):
        with pytest.raises(ValueError):
            encode_pauli_string("AXYZ", 4)


class TestDatasetGenerator:
    def test_vqe_energy_is_real(self):
        terms = h2_hamiltonian_sto3g()
        params = np.zeros(4)
        energy = evaluate_vqe_energy(terms, params, 4)
        assert isinstance(energy, float)

    def test_generate_dataset_shapes(self):
        dataset = generate_dataset(n_qubits=4, n_samples=5, n_pauli_terms=8, n_params=4, max_hamiltonian_terms=16)
        assert dataset["X"].shape == (5, 16 * 17)
        assert dataset["y"].shape == (5, 4)
        assert dataset["E"].shape == (5,)


class TestParameterTransformer:
    def setup_method(self):
        self.model = ParameterTransformer(d_token=17, d_model=16, n_heads=2, n_params=4, seq_len=32)

    def test_forward_shape(self):
        x = np.random.randn(32, 17).astype(np.float32)
        pred = self.model.forward(x)
        assert pred.shape == (4,)

    def test_forward_from_flat(self):
        x = np.random.randn(32 * 17).astype(np.float32)
        pred = self.model.forward(x)
        assert pred.shape == (4,)

    def test_n_parameters_positive(self):
        assert self.model.n_parameters() > 0

    def test_different_inputs_give_different_outputs(self):
        x1 = np.random.randn(32, 17).astype(np.float32)
        x2 = np.random.randn(32, 17).astype(np.float32)
        p1 = self.model.forward(x1)
        p2 = self.model.forward(x2)
        assert not np.allclose(p1, p2)


class TestBaselineVQE:
    def test_baseline_vqe_returns_dict(self):
        terms = h2_hamiltonian_sto3g()
        result = run_baseline_vqe(terms, 4, rng_seed=42)
        for key in ["params", "energy", "history", "converged_at", "init_type"]:
            assert key in result

    def test_baseline_energy_is_finite(self):
        terms = h2_hamiltonian_sto3g()
        result = run_baseline_vqe(terms, 4, rng_seed=0, n_iters=5)
        assert np.isfinite(result["energy"])

    def test_energy_decreases(self):
        terms = h2_hamiltonian_sto3g()
        result = run_baseline_vqe(terms, 4, rng_seed=0, n_iters=20, lr=0.1)
        # Energy should generally decrease over optimization
        assert result["history"][-1] <= result["history"][0] + 0.1  # allow small tolerance
