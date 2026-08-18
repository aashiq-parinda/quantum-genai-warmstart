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
    single_qubit_states = []
    for i in range(n_qubits):
        theta = params[i] if i < len(params) else 0.0
        psi_i = np.array([np.cos(theta / 2), np.sin(theta / 2)], dtype=complex)
        single_qubit_states.append(psi_i)

    # Build full statevector |ψ⟩ via tensor product
    psi = single_qubit_states[0]
    for i in range(1, n_qubits):
        psi = np.kron(psi, single_qubit_states[i])

    Pauli = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

    energy = 0.0
    for pauli_str, coeff in pauli_terms:
        # Pad pauli_str with 'I' up to n_qubits if shorter
        if len(pauli_str) < n_qubits:
            pauli_str = pauli_str + "I" * (n_qubits - len(pauli_str))
        
        # Build Pauli tensor P_k
        P = Pauli[pauli_str[0].upper()]
        for ch in pauli_str[1:n_qubits]:
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
    """Generate synthetic VQE training dataset."""
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


def generate_molecular_dataset(
    n_max_qubits: int = 8,
    max_hamiltonian_terms: int = 64,
    rng_seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Generate structured multi-molecule dataset split into Train, Interpolation, and Held-Out OOD.

    Returns
    -------
    dict with keys 'train', 'val_interpolation', 'test_ood'
    """
    from qwarmstart.data.hamiltonian_encoder import (
        h2_hamiltonian_sto3g, lih_hamiltonian_sto3g, beh2_hamiltonian_sto3g, h4_chain_hamiltonian
    )

    d_token = n_max_qubits * 4 + 1
    d_flat = max_hamiltonian_terms * d_token

    # 1. Train Set (H2, LiH at select bond lengths + synthetic random Hamiltonians)
    train_specs = [
        ("H2", 4, h2_hamiltonian_sto3g(r)) for r in [0.5, 0.7, 0.9, 1.1, 1.5, 2.0]
    ] + [
        ("LiH", 6, lih_hamiltonian_sto3g(r)) for r in [1.0, 1.3, 1.6, 2.0, 2.5]
    ]

    # Add random 4-qubit and 6-qubit Hamiltonians
    rng = np.random.default_rng(rng_seed)
    for _ in range(40):
        nq = int(rng.choice([4, 6]))
        train_specs.append(("Random", nq, random_hamiltonian(nq, 12, rng_seed=int(rng.integers(0, 1000000)))))

    X_train = np.zeros((len(train_specs), d_flat), dtype=np.float32)
    y_train = np.zeros((len(train_specs), n_max_qubits), dtype=np.float32)
    E_train = np.zeros(len(train_specs), dtype=np.float32)
    meta_train = []

    for i, (name, nq, terms) in enumerate(train_specs):
        X_train[i] = hamiltonian_to_flat_vector(terms, n_max_qubits, max_hamiltonian_terms)
        opt_params, opt_e = run_vqe_optimization(terms, nq, rng_seed=i)
        y_train[i, :nq] = opt_params
        E_train[i] = opt_e
        meta_train.append({"molecule": name, "n_qubits": nq})

    # 2. Validation Set: Interpolation on unseen bond lengths of H2 and LiH
    val_specs = [
        ("H2", 4, h2_hamiltonian_sto3g(r), r) for r in [0.735, 1.3, 1.8]
    ] + [
        ("LiH", 6, lih_hamiltonian_sto3g(r), r) for r in [1.4, 1.8, 2.2]
    ]

    X_val = np.zeros((len(val_specs), d_flat), dtype=np.float32)
    y_val = np.zeros((len(val_specs), n_max_qubits), dtype=np.float32)
    E_val = np.zeros(len(val_specs), dtype=np.float32)
    meta_val = []

    for i, (name, nq, terms, r) in enumerate(val_specs):
        X_val[i] = hamiltonian_to_flat_vector(terms, n_max_qubits, max_hamiltonian_terms)
        opt_params, opt_e = run_vqe_optimization(terms, nq, rng_seed=100 + i)
        y_val[i, :nq] = opt_params
        E_val[i] = opt_e
        meta_val.append({"molecule": name, "n_qubits": nq, "bond_length": r, "terms": terms})

    # 3. Test Set: Held-Out Out-of-Distribution (BeH2 6-qubit, H4 chain 8-qubit)
    test_specs = [
        ("BeH2", 6, beh2_hamiltonian_sto3g(r), r) for r in [1.0, 1.3, 1.6, 2.0, 2.5]
    ] + [
        ("H4", 8, h4_chain_hamiltonian(r), r) for r in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    ]

    X_test = np.zeros((len(test_specs), d_flat), dtype=np.float32)
    y_test = np.zeros((len(test_specs), n_max_qubits), dtype=np.float32)
    E_test = np.zeros(len(test_specs), dtype=np.float32)
    meta_test = []

    for i, (name, nq, terms, r) in enumerate(test_specs):
        X_test[i] = hamiltonian_to_flat_vector(terms, n_max_qubits, max_hamiltonian_terms)
        opt_params, opt_e = run_vqe_optimization(terms, nq, rng_seed=200 + i)
        y_test[i, :nq] = opt_params
        E_test[i] = opt_e
        meta_test.append({"molecule": name, "n_qubits": nq, "bond_length": r, "terms": terms})

    return {
        "train": {"X": X_train, "y": y_train, "E": E_train, "meta": meta_train},
        "val_interpolation": {"X": X_val, "y": y_val, "E": E_val, "meta": meta_val},
        "test_ood": {"X": X_test, "y": y_test, "E": E_test, "meta": meta_test},
    }

