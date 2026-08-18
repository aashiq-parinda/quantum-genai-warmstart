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
    max_terms: int = 64,
    n_max_qubits: int = 8,
    rng_seed: int = 0,
) -> Dict[str, Any]:
    """Compare random-init vs warm-start VQE on a single Hamiltonian."""
    # Baseline: random init
    baseline = run_baseline_vqe(pauli_terms, n_qubits, rng_seed=rng_seed)

    # Warm-start: transformer predicted init
    h_vec = hamiltonian_to_flat_vector(pauli_terms, n_max_qubits, max_terms)
    warmstart_params_full = model.forward(h_vec)
    warmstart_params = warmstart_params_full[:n_qubits]
    warmstart = run_vqe_from_init(pauli_terms, n_qubits, warmstart_params)
    warmstart["init_type"] = "transformer"

    iter_reduction = 1.0 - (warmstart["converged_at"] / max(baseline["converged_at"], 1))
    energy_diff = abs(warmstart["energy"] - baseline["energy"])

    return {
        "n_qubits": n_qubits,
        "baseline_iters": baseline["converged_at"],
        "warmstart_iters": warmstart["converged_at"],
        "iter_reduction": iter_reduction,
        "baseline_energy": baseline["energy"],
        "warmstart_energy": warmstart["energy"],
        "energy_diff_hartree": energy_diff,
        "hypothesis_iter_met": iter_reduction >= HYPOTHESIS_THRESHOLD_ITER_REDUCTION,
        "hypothesis_energy_met": energy_diff <= HYPOTHESIS_THRESHOLD_ENERGY_mHA,
    }


def evaluate_molecular_suite(
    model: ParameterTransformer,
    dataset_split: List[Dict[str, Any]],
    max_terms: int = 64,
    n_max_qubits: int = 8,
) -> Dict[str, Any]:
    """Evaluate ParameterTransformer model across a list of molecular test metadata dicts."""
    results = []
    for i, meta in enumerate(dataset_split):
        res = evaluate_single_hamiltonian(
            model,
            meta["terms"],
            meta["n_qubits"],
            max_terms=max_terms,
            n_max_qubits=n_max_qubits,
            rng_seed=i * 10 + 7,
        )
        res["molecule"] = meta["molecule"]
        res["bond_length"] = meta.get("bond_length", None)
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
        "n_samples": len(results),
        "avg_iter_reduction": avg_iter_reduction,
        "avg_energy_diff_hartree": avg_energy_diff,
        "pct_iter_hypothesis_met": pct_iter_met,
        "pct_energy_hypothesis_met": pct_energy_met,
        "hypothesis_supported": hypothesis_supported,
        "results": results,
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
        res = evaluate_single_hamiltonian(
            model, pauli_terms, n_qubits, max_terms=max_terms, n_max_qubits=n_qubits, rng_seed=i
        )
        results.append(res)

from scipy import stats


def evaluate_single_hamiltonian_multi_seed(
    model: ParameterTransformer,
    pauli_terms: List[Tuple[str, float]],
    n_qubits: int,
    molecule_name: str = None,
    n_seeds: int = 10,
    max_terms: int = 64,
    n_max_qubits: int = 8,
) -> Dict[str, Any]:
    """Run warm-start vs random-init AND Hartree-Fock baselines across N seeds with statistical testing.

    Parameters
    ----------
    model : ParameterTransformer
    pauli_terms : Hamiltonian terms
    n_qubits : int
    molecule_name : str, optional
    n_seeds : int — number of random seeds (default 10)
    max_terms : int
    n_max_qubits : int

    Returns
    -------
    dict with:
      - 'energy_mean_base', 'energy_std_base'
      - 'energy_mean_hf', 'energy_std_hf'
      - 'energy_mean_warm', 'energy_std_warm'
      - 'iter_mean_base', 'iter_mean_hf', 'iter_mean_warm'
      - 'p_value_warm_vs_base', 'p_value_warm_vs_hf'
      - 'beats_hartree_fock': bool
    """
    from qwarmstart.models.baseline_vqe import run_hartree_fock_vqe

    h_vec = hamiltonian_to_flat_vector(pauli_terms, n_max_qubits, max_terms)
    warmstart_params_full = model.forward(h_vec)
    warmstart_params = warmstart_params_full[:n_qubits]

    base_energies = []
    base_iters = []
    hf_energies = []
    hf_iters = []
    warm_energies = []
    warm_iters = []

    for seed in range(n_seeds):
        # 1. Random baseline
        baseline = run_baseline_vqe(pauli_terms, n_qubits, rng_seed=seed)
        base_energies.append(baseline["energy"])
        base_iters.append(baseline["converged_at"])

        # 2. Hartree-Fock baseline (with small noise perturbation across seeds)
        rng = np.random.default_rng(seed)
        hf_init = run_hartree_fock_vqe(pauli_terms, n_qubits, molecule_name=molecule_name)
        hf_noisy_params = hf_init["params"] + (rng.normal(0, 0.01, size=n_qubits) if seed > 0 else np.zeros(n_qubits))
        hf_run = run_vqe_from_init(pauli_terms, n_qubits, hf_noisy_params)
        hf_energies.append(hf_run["energy"])
        hf_iters.append(hf_run["converged_at"])

        # 3. Transformer warm-start
        warm_noise = rng.normal(0, 0.01, size=n_qubits) if seed > 0 else np.zeros(n_qubits)
        warm_init = warmstart_params + warm_noise
        warmstart = run_vqe_from_init(pauli_terms, n_qubits, warm_init)
        warm_energies.append(warmstart["energy"])
        warm_iters.append(warmstart["converged_at"])

    base_energies = np.array(base_energies, dtype=np.float64)
    hf_energies = np.array(hf_energies, dtype=np.float64)
    warm_energies = np.array(warm_energies, dtype=np.float64)
    base_iters = np.array(base_iters, dtype=np.float64)
    hf_iters = np.array(hf_iters, dtype=np.float64)
    warm_iters = np.array(warm_iters, dtype=np.float64)

    # Paired t-tests
    _, p_val_warm_vs_base = stats.ttest_rel(warm_energies, base_energies) if not np.allclose(warm_energies, base_energies) else (0, 1.0)
    _, p_val_warm_vs_hf = stats.ttest_rel(warm_energies, hf_energies) if not np.allclose(warm_energies, hf_energies) else (0, 1.0)
    _, p_val_hf_vs_base = stats.ttest_rel(hf_energies, base_energies) if not np.allclose(hf_energies, base_energies) else (0, 1.0)

    iter_reduction_vs_base = 1.0 - (float(np.mean(warm_iters)) / max(float(np.mean(base_iters)), 1.0))
    iter_reduction_vs_hf = 1.0 - (float(np.mean(warm_iters)) / max(float(np.mean(hf_iters)), 1.0))

    beats_hf_energy = float(np.mean(warm_energies)) <= float(np.mean(hf_energies)) + 0.005
    beats_hf_iters = float(np.mean(warm_iters)) <= float(np.mean(hf_iters))
    beats_hf = beats_hf_energy and beats_hf_iters

    return {
        "n_qubits": n_qubits,
        "n_seeds": n_seeds,
        "energy_mean_base": float(np.mean(base_energies)),
        "energy_std_base": float(np.std(base_energies)),
        "energy_mean_hf": float(np.mean(hf_energies)),
        "energy_std_hf": float(np.std(hf_energies)),
        "energy_mean_warm": float(np.mean(warm_energies)),
        "energy_std_warm": float(np.std(warm_energies)),
        "iter_mean_base": float(np.mean(base_iters)),
        "iter_std_base": float(np.std(base_iters)),
        "iter_mean_hf": float(np.mean(hf_iters)),
        "iter_std_hf": float(np.std(hf_iters)),
        "iter_mean_warm": float(np.mean(warm_iters)),
        "iter_std_warm": float(np.std(warm_iters)),
        "iter_reduction_mean": iter_reduction_vs_base,
        "iter_reduction_vs_hf": iter_reduction_vs_hf,
        "p_value_ttest": float(p_val_warm_vs_base),
        "p_value_warm_vs_hf": float(p_val_warm_vs_hf),
        "p_value_hf_vs_base": float(p_val_hf_vs_base),
        "statistically_significant": float(p_val_warm_vs_base) < 0.05,
        "beats_hartree_fock": beats_hf,
    }


def evaluate_molecular_suite_multi_seed(
    model: ParameterTransformer,
    dataset_split: List[Dict[str, Any]],
    n_seeds: int = 10,
    max_terms: int = 64,
    n_max_qubits: int = 8,
) -> Dict[str, Any]:
    """Evaluate multi-seed performance and significance across a molecular suite."""
    results = []
    for meta in dataset_split:
        res = evaluate_single_hamiltonian_multi_seed(
            model,
            meta["terms"],
            meta["n_qubits"],
            molecule_name=meta.get("molecule", None),
            n_seeds=n_seeds,
            max_terms=max_terms,
            n_max_qubits=n_max_qubits,
        )
        res["molecule"] = meta["molecule"]
        res["bond_length"] = meta.get("bond_length", None)
        results.append(res)

    avg_iter_reduction = float(np.mean([r["iter_reduction_mean"] for r in results]))
    avg_iter_reduction_vs_hf = float(np.mean([r["iter_reduction_vs_hf"] for r in results]))
    avg_energy_warm = float(np.mean([r["energy_mean_warm"] for r in results]))
    avg_energy_base = float(np.mean([r["energy_mean_base"] for r in results]))
    avg_energy_hf = float(np.mean([r["energy_mean_hf"] for r in results]))
    pct_beats_hf = float(np.mean([1.0 if r["beats_hartree_fock"] else 0.0 for r in results])) * 100

    return {
        "n_samples": len(results),
        "n_seeds": n_seeds,
        "avg_iter_reduction_vs_random": avg_iter_reduction,
        "avg_iter_reduction_vs_hf": avg_iter_reduction_vs_hf,
        "avg_energy_warm": avg_energy_warm,
        "avg_energy_base": avg_energy_base,
        "avg_energy_hf": avg_energy_hf,
        "pct_beats_hartree_fock": pct_beats_hf,
        "results": results,
    }


def compute_parameter_gradients(
    pauli_terms: List[Tuple[str, float]],
    params: np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    """Compute exact Parameter-Shift gradients ∂E/∂θ_k for a given parameter vector θ."""
    grad = np.zeros_like(params)
    for p_idx in range(n_qubits):
        p_plus = params.copy(); p_plus[p_idx] += np.pi / 2
        p_minus = params.copy(); p_minus[p_idx] -= np.pi / 2
        grad[p_idx] = (
            evaluate_vqe_energy(pauli_terms, p_plus, n_qubits) -
            evaluate_vqe_energy(pauli_terms, p_minus, n_qubits)
        ) / 2.0
    return grad


def measure_gradient_variance(
    pauli_terms: List[Tuple[str, float]],
    n_qubits: int,
    init_type: str = "random",
    model: ParameterTransformer = None,
    molecule_name: str = None,
    n_samples: int = 100,
    sigma: float = 0.1,
    max_terms: int = 64,
    n_max_qubits: int = 8,
    rng_seed: int = 42,
) -> Dict[str, Any]:
    """Measure gradient variance Var[∂E/∂θ_k] across parameter initializations.

    Parameters
    ----------
    pauli_terms : List of (pauli_str, coeff)
    n_qubits : int
    init_type : 'random', 'hartree_fock', or 'transformer'
    model : ParameterTransformer, required if init_type == 'transformer'
    molecule_name : str, optional
    n_samples : int — number of parameter perturbation samples
    sigma : float — Gaussian perturbation std dev for local neighborhood sampling
    max_terms : int
    n_max_qubits : int

    Returns
    -------
    dict with:
      - 'grad_variance': float — mean variance of gradients Var[∂E/∂θ]
      - 'grad_magnitude_mean': float — mean L2 norm of gradient vector
      - 'per_param_variance': np.ndarray — variance per parameter component
    """
    from qwarmstart.models.baseline_vqe import get_hartree_fock_params

    rng = np.random.default_rng(rng_seed)
    gradients = []

    if init_type == "random":
        for _ in range(n_samples):
            p = rng.uniform(0, 2 * np.pi, n_qubits)
            g = compute_parameter_gradients(pauli_terms, p, n_qubits)
            gradients.append(g)

    elif init_type == "hartree_fock":
        base_p = get_hartree_fock_params(n_qubits, molecule_name=molecule_name)
        for _ in range(n_samples):
            p = base_p + rng.normal(0, sigma, size=n_qubits)
            g = compute_parameter_gradients(pauli_terms, p, n_qubits)
            gradients.append(g)

    elif init_type == "transformer":
        if model is None:
            raise ValueError("Model required for transformer init_type")
        h_vec = hamiltonian_to_flat_vector(pauli_terms, n_max_qubits, max_terms)
        base_p = model.forward(h_vec)[:n_qubits]
        for _ in range(n_samples):
            p = base_p + rng.normal(0, sigma, size=n_qubits)
            g = compute_parameter_gradients(pauli_terms, p, n_qubits)
            gradients.append(g)

    else:
        raise ValueError(f"Unknown init_type: {init_type}")

    gradients = np.array(gradients, dtype=np.float64)  # shape (n_samples, n_qubits)
    per_param_var = np.var(gradients, axis=0)
    mean_var = float(np.mean(per_param_var))
    grad_norms = np.linalg.norm(gradients, axis=1)
    mean_norm = float(np.mean(grad_norms))

    return {
        "init_type": init_type,
        "n_qubits": n_qubits,
        "grad_variance": mean_var,
        "grad_magnitude_mean": mean_norm,
        "per_param_variance": per_param_var,
    }


def run_barren_plateau_diagnostic(
    model: ParameterTransformer = None,
    n_samples: int = 100,
    max_terms: int = 64,
    n_max_qubits: int = 8,
) -> Dict[str, Any]:
    """Run comprehensive Barren Plateau Diagnostic comparing gradient variance vs qubit count.

    Evaluates Var[∂E/∂θ] for Random Init, Hartree-Fock Init, and Transformer Warm-Start
    across 4-qubit (H2), 6-qubit (LiH, BeH2), and 8-qubit (H4) systems.
    """
    from qwarmstart.data.hamiltonian_encoder import (
        h2_hamiltonian_sto3g, lih_hamiltonian_sto3g, beh2_hamiltonian_sto3g, h4_chain_hamiltonian
    )

    test_systems = [
        ("H2 (In-Dist)", 4, h2_hamiltonian_sto3g(0.735), "H2"),
        ("LiH (In-Dist)", 6, lih_hamiltonian_sto3g(1.6), "LiH"),
        ("BeH2 (Zero-Shot OOD)", 6, beh2_hamiltonian_sto3g(1.3), "BeH2"),
        ("H4 (Zero-Shot OOD)", 8, h4_chain_hamiltonian(1.0), "H4"),
    ]

    diagnostic_results = []

    for name, nq, terms, mol in test_systems:
        res_rand = measure_gradient_variance(terms, nq, "random", n_samples=n_samples)
        res_hf = measure_gradient_variance(terms, nq, "hartree_fock", molecule_name=mol, n_samples=n_samples)
        
        sys_dict = {
            "system": name,
            "n_qubits": nq,
            "var_random": res_rand["grad_variance"],
            "var_hf": res_hf["grad_variance"],
            "norm_random": res_rand["grad_magnitude_mean"],
            "norm_hf": res_hf["grad_magnitude_mean"],
        }

        if model is not None:
            res_trans = measure_gradient_variance(
                terms, nq, "transformer", model=model, molecule_name=mol,
                n_samples=n_samples, max_terms=max_terms, n_max_qubits=n_max_qubits
            )
            sys_dict["var_transformer"] = res_trans["grad_variance"]
            sys_dict["norm_transformer"] = res_trans["grad_magnitude_mean"]
            sys_dict["ratio_trans_vs_random"] = res_trans["grad_variance"] / max(res_rand["grad_variance"], 1e-12)
            sys_dict["ratio_trans_vs_hf"] = res_trans["grad_variance"] / max(res_hf["grad_variance"], 1e-12)

        diagnostic_results.append(sys_dict)

    return {
        "n_samples": n_samples,
        "results": diagnostic_results,
    }




