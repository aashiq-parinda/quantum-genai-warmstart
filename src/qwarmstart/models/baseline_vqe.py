"""Baseline VQE with Random Parameter Initialization.

Standard VQE with no smart initialization — used as the baseline
to evaluate whether transformer warm-starting reduces convergence costs.

Runs VQE from random θ_0 ~ Uniform[0, 2π]^n and tracks:
  - Iterations to convergence
  - Final energy
  - Energy trajectory
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from qwarmstart.data.dataset_generator import evaluate_vqe_energy, evaluate_vqe_energy_circuit


def run_vqe_from_init(
    pauli_terms: List[Tuple[str, float]],
    n_qubits: int,
    initial_params: np.ndarray,
    entangling_pairs: List[Tuple[int, int]] = None,
    n_iters: int = 500,
    lr: float = 0.05,
    tol: float = 1e-5,
) -> Dict[str, Any]:
    """Run VQE from a given initial parameter vector on specified circuit architecture.

    Parameters
    ----------
    pauli_terms : Hamiltonian Pauli terms
    n_qubits : int
    initial_params : np.ndarray — starting parameters
    entangling_pairs : List of (control, target) qubit pairs (default None -> [])
    n_iters : int — maximum iterations
    lr : float — learning rate
    tol : float — convergence tolerance on energy change

    Returns
    -------
    dict with:
      - 'params': final optimized parameters
      - 'energy': final energy
      - 'history': list of energies per iteration
      - 'converged_at': iteration index at convergence
      - 'entangling_pairs': list of pairs used
      - 'n_cx_gates': count of 2-qubit gates
    """
    pairs = entangling_pairs if entangling_pairs is not None else []
    params = initial_params.copy()
    n_p = len(params)
    history = [evaluate_vqe_energy_circuit(pauli_terms, params, pairs, n_qubits)]
    converged_at = n_iters

    for it in range(n_iters):
        grad = np.zeros_like(params)
        for p_idx in range(n_p):
            p_plus = params.copy(); p_plus[p_idx] += np.pi / 2
            p_minus = params.copy(); p_minus[p_idx] -= np.pi / 2
            grad[p_idx] = (
                evaluate_vqe_energy_circuit(pauli_terms, p_plus, pairs, n_qubits) -
                evaluate_vqe_energy_circuit(pauli_terms, p_minus, pairs, n_qubits)
            ) / 2.0
        params -= lr * grad
        energy = evaluate_vqe_energy_circuit(pauli_terms, params, pairs, n_qubits)
        history.append(energy)

        if it > 5 and abs(history[-1] - history[-2]) < tol:
            converged_at = it + 1
            break

    return {
        "params": params,
        "energy": history[-1],
        "history": history,
        "converged_at": converged_at,
        "entangling_pairs": pairs,
        "n_cx_gates": len(pairs),
    }


def run_baseline_vqe(
    pauli_terms: List[Tuple[str, float]],
    n_qubits: int,
    rng_seed: int = 42,
    entangling_pairs: List[Tuple[int, int]] = None,
    n_params: int = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run VQE with random initialization (baseline)."""
    rng = np.random.default_rng(rng_seed)
    p_dim = n_params if n_params is not None else n_qubits
    initial_params = rng.uniform(0, 2 * np.pi, p_dim)
    result = run_vqe_from_init(pauli_terms, n_qubits, initial_params, entangling_pairs=entangling_pairs, **kwargs)
    result["init_type"] = "random"
    result["initial_params"] = initial_params
    return result


def run_fixed_hea_vqe(
    pauli_terms: List[Tuple[str, float]],
    n_qubits: int,
    rng_seed: int = 42,
    n_layers: int = 1,
    **kwargs,
) -> Dict[str, Any]:
    """Run Fixed Linear Nearest-Neighbor Hardware-Efficient Ansatz (HEA) baseline.

    Uses CNOT chain across all adjacent pairs (i, i+1) with random initial parameters.
    """
    pairs = [(i, i + 1) for i in range(n_qubits - 1)]
    # 2*n_qubits parameters for 1 entangling layer (rotation before and after CX)
    p_dim = n_qubits * (n_layers + 1)
    res = run_baseline_vqe(pauli_terms, n_qubits, rng_seed=rng_seed, entangling_pairs=pairs, n_params=p_dim, **kwargs)
    res["init_type"] = "fixed_hea"
    res["circuit_depth"] = n_layers * 2 + 1
    return res


def run_baseline_vqe_multi_seed(
    pauli_terms: List[Tuple[str, float]],
    n_qubits: int,
    seeds: List[int] = list(range(10)),
    entangling_pairs: List[Tuple[int, int]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run VQE across multiple random seeds and calculate summary statistics."""
    runs = [run_baseline_vqe(pauli_terms, n_qubits, rng_seed=s, entangling_pairs=entangling_pairs, **kwargs) for s in seeds]
    energies = np.array([r["energy"] for r in runs], dtype=np.float64)
    iterations = np.array([r["converged_at"] for r in runs], dtype=np.float64)

    return {
        "energies": energies,
        "iterations": iterations,
        "energy_mean": float(np.mean(energies)),
        "energy_std": float(np.std(energies)),
        "iter_mean": float(np.mean(iterations)),
        "iter_std": float(np.std(iterations)),
        "runs": runs,
    }


def get_hartree_fock_params(n_qubits: int, molecule_name: str = None) -> np.ndarray:
    """Generate Hartree-Fock reference state parameters for single-qubit Ry rotation ansatz."""
    if molecule_name == "H2":
        n_elec = 2
    elif molecule_name == "LiH":
        n_elec = 2
    elif molecule_name == "BeH2":
        n_elec = 4
    elif molecule_name == "H4":
        n_elec = 4
    else:
        n_elec = max(1, n_qubits // 2)

    params = np.zeros(n_qubits, dtype=np.float32)
    params[:min(n_elec, n_qubits)] = np.pi
    return params


def run_hartree_fock_vqe(
    pauli_terms: List[Tuple[str, float]],
    n_qubits: int,
    molecule_name: str = None,
    entangling_pairs: List[Tuple[int, int]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run VQE initialized from Hartree-Fock classical parameters."""
    hf_params = get_hartree_fock_params(n_qubits, molecule_name=molecule_name)
    result = run_vqe_from_init(pauli_terms, n_qubits, hf_params, entangling_pairs=entangling_pairs, **kwargs)
    result["init_type"] = "hartree_fock"
    result["initial_params"] = hf_params
    return result


