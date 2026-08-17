"""Benchmark Evaluation: Warm-Start vs Random Initialization.

Tests hypothesis:
  > A transformer-predicted initialization reduces VQE convergence
  > iterations by ≥ 40% without sacrificing energy quality (within 5 mHa).

Experiments:
  E1: Random-init baseline on test Hamiltonians
  E2: Transformer warm-start on same Hamiltonians
  E3: OOD test on unseen Hamiltonians (self-disproof attempt)
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from qwarmstart.data.hamiltonian_encoder import hamiltonian_to_flat_vector, random_hamiltonian
from qwarmstart.data.dataset_generator import evaluate_vqe_energy
from qwarmstart.models.baseline_vqe import run_baseline_vqe, run_vqe_from_init
from qwarmstart.models.parameter_transformer import ParameterTransformer


HYPOTHESIS_THRESHOLD_ITER_REDUCTION = 0.40  # 40% fewer iterations
HYPOTHESIS_THRESHOLD_ENERGY_mHA = 0.005     # 5 mHartree tolerance


def evaluate_single_hamiltonian(
    model: ParameterTransformer,
    pauli_terms: List[Tuple[str, float]],
    n_qubits: int,
    max_terms: int = 32,
    rng_seed: int = 0,
) -> Dict[str, Any]:
    """Compare random-init vs warm-start VQE on a single Hamiltonian."""
    # Baseline: random init
    baseline = run_baseline_vqe(pauli_terms, n_qubits, rng_seed=rng_seed)

    # Warm-start: transformer predicted init
    h_vec = hamiltonian_to_flat_vector(pauli_terms, n_qubits, max_terms)
    warmstart_params = model.forward(h_vec)
    warmstart = run_vqe_from_init(pauli_terms, n_qubits, warmstart_params)
    warmstart["init_type"] = "transformer"

    iter_reduction = 1.0 - (warmstart["converged_at"] / max(baseline["converged_at"], 1))
    energy_diff = abs(warmstart["energy"] - baseline["energy"])

    return {
        "baseline_iters": baseline["converged_at"],
        "warmstart_iters": warmstart["converged_at"],
        "iter_reduction": iter_reduction,
        "baseline_energy": baseline["energy"],
        "warmstart_energy": warmstart["energy"],
        "energy_diff_hartree": energy_diff,
        "hypothesis_iter_met": iter_reduction >= HYPOTHESIS_THRESHOLD_ITER_REDUCTION,
        "hypothesis_energy_met": energy_diff <= HYPOTHESIS_THRESHOLD_ENERGY_mHA,
    }


def run_benchmark(
    model: ParameterTransformer,
    n_qubits: int = 4,
    n_test_hamiltonians: int = 20,
    max_terms: int = 32,
    rng_seed: int = 999,
) -> Dict[str, Any]:
    """Run full warm-start vs random benchmark suite."""
    rng = np.random.default_rng(rng_seed)
    results = []

    for i in range(n_test_hamiltonians):
        pauli_terms = random_hamiltonian(n_qubits, 12, rng_seed=int(rng.integers(100000, 999999)))
        res = evaluate_single_hamiltonian(model, pauli_terms, n_qubits, max_terms, rng_seed=i)
        results.append(res)

    avg_iter_reduction = np.mean([r["iter_reduction"] for r in results])
    avg_energy_diff = np.mean([r["energy_diff_hartree"] for r in results])
    pct_iter_met = np.mean([r["hypothesis_iter_met"] for r in results]) * 100
    pct_energy_met = np.mean([r["hypothesis_energy_met"] for r in results]) * 100

    hypothesis_supported = (
        avg_iter_reduction >= HYPOTHESIS_THRESHOLD_ITER_REDUCTION and
        avg_energy_diff <= HYPOTHESIS_THRESHOLD_ENERGY_mHA
    )

    return {
        "n_hamiltonians_tested": n_test_hamiltonians,
        "avg_iter_reduction": avg_iter_reduction,
        "avg_energy_diff_hartree": avg_energy_diff,
        "pct_iter_hypothesis_met": pct_iter_met,
        "pct_energy_hypothesis_met": pct_energy_met,
        "hypothesis_supported": hypothesis_supported,
        "results": results,
    }
