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

    def test_encode_pauli_string_padding(self):
        enc = encode_pauli_string("IXYZ", 8)
        assert enc.shape == (32,)  # 8 qubits * 4 Pauli

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

    def test_multi_molecule_generators(self):
        from qwarmstart.data.hamiltonian_encoder import lih_hamiltonian_sto3g, beh2_hamiltonian_sto3g, h4_chain_hamiltonian
        assert len(h2_hamiltonian_sto3g(0.735)) > 0
        assert len(lih_hamiltonian_sto3g(1.6)) > 0
        assert len(beh2_hamiltonian_sto3g(1.3)) > 0
        assert len(h4_chain_hamiltonian(1.0)) > 0

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

    def test_generate_molecular_dataset_splits(self):
        from qwarmstart.data.dataset_generator import generate_molecular_dataset
        mol_data = generate_molecular_dataset(n_max_qubits=8, max_hamiltonian_terms=32)
        assert "train" in mol_data
        assert "val_interpolation" in mol_data
        assert "test_ood" in mol_data
        assert mol_data["train"]["X"].shape[1] == 32 * (8 * 4 + 1)
        assert len(mol_data["test_ood"]["meta"]) > 0


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

    def test_multi_seed_baseline_stats(self):
        from qwarmstart.models.baseline_vqe import run_baseline_vqe_multi_seed
        terms = h2_hamiltonian_sto3g()
        res = run_baseline_vqe_multi_seed(terms, 4, seeds=[0, 1, 2], n_iters=10)
        assert len(res["energies"]) == 3
        assert "energy_mean" in res
        assert "energy_std" in res


    def test_hartree_fock_params(self):
        from qwarmstart.models.baseline_vqe import get_hartree_fock_params, run_hartree_fock_vqe
        hf_h2 = get_hartree_fock_params(4, "H2")
        assert np.allclose(hf_h2, [np.pi, np.pi, 0, 0])
        hf_beh2 = get_hartree_fock_params(6, "BeH2")
        assert np.allclose(hf_beh2, [np.pi, np.pi, np.pi, np.pi, 0, 0])


        terms = h2_hamiltonian_sto3g()
        res = run_hartree_fock_vqe(terms, 4, molecule_name="H2", n_iters=10)
        assert res["init_type"] == "hartree_fock"
        assert np.isfinite(res["energy"])


class TestEvaluation:
    def test_multi_seed_evaluation_ttest(self):
        from qwarmstart.benchmarks.evaluation import evaluate_single_hamiltonian_multi_seed
        model = ParameterTransformer(d_token=33, d_model=16, n_heads=2, n_params=8, seq_len=32)
        terms = h2_hamiltonian_sto3g()
        res = evaluate_single_hamiltonian_multi_seed(model, terms, n_qubits=4, molecule_name="H2", n_seeds=5, max_terms=32, n_max_qubits=8)
        assert "p_value_ttest" in res
        assert "energy_mean_base" in res
        assert "energy_mean_hf" in res
        assert "energy_mean_warm" in res
        assert "beats_hartree_fock" in res
        assert isinstance(res["statistically_significant"], bool)

    def test_barren_plateau_diagnostic(self):
        from qwarmstart.benchmarks.evaluation import run_barren_plateau_diagnostic, measure_gradient_variance
        model = ParameterTransformer(d_token=33, d_model=16, n_heads=2, n_params=8, seq_len=32)
        terms = h2_hamiltonian_sto3g()
        res_rand = measure_gradient_variance(terms, 4, "random", n_samples=10)
        assert "grad_variance" in res_rand
        assert res_rand["grad_variance"] >= 0.0

        diag = run_barren_plateau_diagnostic(model=model, n_samples=10, max_terms=32, n_max_qubits=8)
        assert "results" in diag
        assert len(diag["results"]) == 4
        assert "ratio_trans_vs_random" in diag["results"][0]




