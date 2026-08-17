"""VQE Training Dataset Generator.

Generates {Hamiltonian embedding → optimal VQE parameters} pairs
by running VQE-style energy minimization on random Hamiltonians and
recording the converged parameter values.

Dataset schema:
    X[i] = hamiltonian_flat_vector     shape (max_terms * d_token,)
    y[i] = optimal_theta               shape (n_params,)
    E[i] = optimal_energy              scalar
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from qwarmstart.data.hamiltonian_encoder import (
    hamiltonian_to_flat_vector,
    random_hamiltonian,
    h2_hamiltonian_sto3g,
    lih_hamiltonian_sto3g,
)


def evaluate_vqe_energy(
    pauli_terms: List[Tuple[str, float]],
    params: np.ndarray,
    n_qubits: int,
) -> float:
    """Evaluate VQE energy expectation E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩.

    Uses Ry(θ_i) single-qubit ansatz on each qubit.
    Simple but physically motivated for demonstration.

    E(θ) = Σ_k h_k ⟨ψ(θ)|P_k|ψ(θ)⟩

    Parameters
    ----------
    pauli_terms : list of (pauli_str, coefficient)
    params : np.ndarray shape (n_qubits,) — Ry rotation angles
    n_qubits : int

    Returns
    -------
    float — energy expectation value
    """
    # Build |ψ(θ)⟩ = ⊗_i Ry(θ_i)|0⟩
    # Single-qubit state: Ry(θ)|0⟩ = [cos(θ/2), sin(θ/2)]
    single_qubit_states = []
    for i in range(n_qubits):
        theta = params[i] if i < len(params) else 0.0
        psi_i = np.array([np.cos(theta / 2), np.sin(theta / 2)], dtype=complex)
        single_qubit_states.append(psi_i)

    # Build full statevector |ψ⟩ via tensor product
    psi = single_qubit_states[0]
    for i in range(1, n_qubits):
        psi = np.kron(psi, single_qubit_states[i])

    dim = 2 ** n_qubits
    Pauli = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

    energy = 0.0
    for pauli_str, coeff in pauli_terms:
        # Build Pauli tensor P_k
        P = Pauli[pauli_str[0].upper()]
        for ch in pauli_str[1:]:
            P = np.kron(P, Pauli[ch.upper()])
        energy += coeff * float(np.real(psi.conj() @ P @ psi))

    return energy


def run_vqe_optimization(
    pauli_terms: List[Tuple[str, float]],
    n_qubits: int,
    n_iters: int = 200,
    lr: float = 0.05,
    rng_seed: int = 0,
) -> Tuple[np.ndarray, float]:
    """Run VQE optimization with Parameter-Shift gradient descent.

    Returns
    -------
    (optimal_params, optimal_energy)
    """
    rng = np.random.default_rng(rng_seed)
    params = rng.uniform(0, 2 * np.pi, n_qubits)

    for _ in range(n_iters):
        grad = np.zeros_like(params)
        for p_idx in range(n_qubits):
            p_plus = params.copy(); p_plus[p_idx] += np.pi / 2
            p_minus = params.copy(); p_minus[p_idx] -= np.pi / 2
            grad[p_idx] = (
                evaluate_vqe_energy(pauli_terms, p_plus, n_qubits) -
                evaluate_vqe_energy(pauli_terms, p_minus, n_qubits)
            ) / 2.0
        params -= lr * grad

    energy = evaluate_vqe_energy(pauli_terms, params, n_qubits)
    return params, energy


def generate_dataset(
    n_qubits: int = 4,
    n_samples: int = 200,
    n_pauli_terms: int = 12,
    n_params: int = 4,
    max_hamiltonian_terms: int = 32,
    rng_seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate synthetic VQE training dataset.

    Parameters
    ----------
    n_qubits : int
    n_samples : int — number of (Hamiltonian, optimal_params) pairs
    n_pauli_terms : int — Pauli terms per random Hamiltonian
    n_params : int — VQE parameter dimension
    max_hamiltonian_terms : int — token padding length

    Returns
    -------
    dict with 'X' (features), 'y' (optimal params), 'E' (energies)
    """
    rng = np.random.default_rng(rng_seed)
    d_token = n_qubits * 4 + 1
    d_flat = max_hamiltonian_terms * d_token

    X = np.zeros((n_samples, d_flat), dtype=np.float32)
    y = np.zeros((n_samples, n_params), dtype=np.float32)
    E = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        pauli_terms = random_hamiltonian(n_qubits, n_pauli_terms, rng_seed=int(rng.integers(0, 1000000)))
        X[i] = hamiltonian_to_flat_vector(pauli_terms, n_qubits, max_hamiltonian_terms)
        opt_params, opt_energy = run_vqe_optimization(pauli_terms, n_qubits, rng_seed=int(rng.integers(0, 10000)))
        y[i] = opt_params[:n_params]
        E[i] = opt_energy

    return {"X": X, "y": y, "E": E}
