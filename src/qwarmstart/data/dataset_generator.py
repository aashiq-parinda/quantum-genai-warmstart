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


def evaluate_vqe_energy_circuit(
    pauli_terms: List[Tuple[str, float]],
    params: np.ndarray,
    entangling_pairs: List[Tuple[int, int]],
    n_qubits: int,
) -> float:
    """Evaluate VQE energy expectation E(θ, E_pairs) = ⟨ψ(θ, E_pairs)|H|ψ(θ, E_pairs)⟩.

    Ansatz:
      1. Layer 1 single-qubit Ry(θ_i) rotations for i in range(n_qubits).
      2. 2-qubit CNOT gates on specified entangling_pairs.
      3. Layer 2 single-qubit Ry(θ_{n_qubits + i}) rotations if len(params) >= 2*n_qubits.

    Parameters
    ----------
    pauli_terms : list of (pauli_str, coefficient)
    params : np.ndarray shape (n_qubits,) or (2*n_qubits,)
    entangling_pairs : list of (control, target) qubit pairs
    n_qubits : int

    Returns
    -------
    float — energy expectation value
    """
    psi = np.zeros(2**n_qubits, dtype=complex)
    psi[0] = 1.0

    # Layer 1: Ry rotations
    for q in range(n_qubits):
        th = params[q] if q < len(params) else 0.0
        Ry = np.array([[np.cos(th / 2), -np.sin(th / 2)], [np.sin(th / 2), np.cos(th / 2)]], dtype=complex)
        psi_tensor = psi.reshape([2] * n_qubits)
        psi_tensor = np.tensordot(Ry, psi_tensor, axes=([1], [q]))
        psi = np.moveaxis(psi_tensor, 0, q).reshape(-1)

    # Layer 2: Entangling CNOT gates
    CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex).reshape(2, 2, 2, 2)
    for c, t in entangling_pairs:
        if c >= n_qubits or t >= n_qubits or c == t:
            continue
        psi_tensor = psi.reshape([2] * n_qubits)
        if c < t:
            out = np.tensordot(CX, psi_tensor, axes=([2, 3], [c, t]))
            psi = np.moveaxis(out, [0, 1], [c, t]).reshape(-1)
        else:
            CX_tc = np.moveaxis(CX, [0, 1, 2, 3], [1, 0, 3, 2])
            out = np.tensordot(CX_tc, psi_tensor, axes=([2, 3], [t, c]))
            psi = np.moveaxis(out, [0, 1], [t, c]).reshape(-1)

    # Layer 3: Ry rotations after entanglement if 2*n_qubits params provided
    if len(params) >= 2 * n_qubits:
        for q in range(n_qubits):
            th = params[n_qubits + q]
            Ry = np.array([[np.cos(th / 2), -np.sin(th / 2)], [np.sin(th / 2), np.cos(th / 2)]], dtype=complex)
            psi_tensor = psi.reshape([2] * n_qubits)
            psi_tensor = np.tensordot(Ry, psi_tensor, axes=([1], [q]))
            psi = np.moveaxis(psi_tensor, 0, q).reshape(-1)

    # Compute energy expectation <psi|H|psi>
    Pauli_dict = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

    energy = 0.0
    for pauli_str, coeff in pauli_terms:
        p = pauli_str + "I" * max(0, n_qubits - len(pauli_str))
        P_psi = psi.copy()
        for q in range(n_qubits):
            ch = p[q].upper()
            if ch != "I":
                op = Pauli_dict[ch]
                P_tensor = P_psi.reshape([2] * n_qubits)
                P_tensor = np.tensordot(op, P_tensor, axes=([1], [q]))
                P_psi = np.moveaxis(P_tensor, 0, q).reshape(-1)
        energy += coeff * float(np.real(np.vdot(psi, P_psi)))

    return energy


def evaluate_vqe_energy(
    pauli_terms: List[Tuple[str, float]],
    params: np.ndarray,
    n_qubits: int,
) -> float:
    """Evaluate VQE energy expectation for product Ry state (0 entangling gates)."""
    return evaluate_vqe_energy_circuit(pauli_terms, params, entangling_pairs=[], n_qubits=n_qubits)


def hamiltonian_to_target_mask(
    pauli_terms: List[Tuple[str, float]],
    n_max_qubits: int = 8,
) -> np.ndarray:
    """Extract ground-truth 2-body interaction binary mask for candidate pairs."""
    from qwarmstart.models.parameter_transformer import get_candidate_pairs
    candidate_pairs = get_candidate_pairs(n_max_qubits)
    mask = np.zeros(len(candidate_pairs), dtype=np.float32)

    interacting_pairs = set()
    for p_str, coeff in pauli_terms:
        active = [i for i, ch in enumerate(p_str) if ch.upper() != "I"]
        if len(active) == 2:
            i, j = sorted(active)
            interacting_pairs.add((i, j))

    for idx, (i, j) in enumerate(candidate_pairs):
        if (i, j) in interacting_pairs:
            mask[idx] = 1.0

    return mask


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
    n_random: int = 15,
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
    for _ in range(n_random):
        nq = int(rng.choice([4, 6]))
        train_specs.append(("Random", nq, random_hamiltonian(nq, 12, rng_seed=int(rng.integers(0, 1000000)))))

    from qwarmstart.models.parameter_transformer import get_candidate_pairs
    candidate_pairs = get_candidate_pairs(n_max_qubits)
    n_pairs = len(candidate_pairs)

    X_train = np.zeros((len(train_specs), d_flat), dtype=np.float32)
    y_train = np.zeros((len(train_specs), n_max_qubits * 2), dtype=np.float32)
    mask_train = np.zeros((len(train_specs), n_pairs), dtype=np.float32)
    E_train = np.zeros(len(train_specs), dtype=np.float32)
    meta_train = []

    for i, (name, nq, terms) in enumerate(train_specs):
        X_train[i] = hamiltonian_to_flat_vector(terms, n_max_qubits, max_hamiltonian_terms)
        mask_train[i] = hamiltonian_to_target_mask(terms, n_max_qubits)
        opt_params, opt_e = run_vqe_optimization(terms, nq, rng_seed=i)
        y_train[i, :nq] = opt_params
        # duplicate/expand for 2-layer parameters
        y_train[i, n_max_qubits: n_max_qubits + nq] = opt_params
        E_train[i] = opt_e
        meta_train.append({"molecule": name, "n_qubits": nq, "terms": terms})

    # 2. Validation Set: Interpolation on unseen bond lengths of H2 and LiH
    val_specs = [
        ("H2", 4, h2_hamiltonian_sto3g(r), r) for r in [0.735, 1.3, 1.8]
    ] + [
        ("LiH", 6, lih_hamiltonian_sto3g(r), r) for r in [1.4, 1.8, 2.2]
    ]

    X_val = np.zeros((len(val_specs), d_flat), dtype=np.float32)
    y_val = np.zeros((len(val_specs), n_max_qubits * 2), dtype=np.float32)
    mask_val = np.zeros((len(val_specs), n_pairs), dtype=np.float32)
    E_val = np.zeros(len(val_specs), dtype=np.float32)
    meta_val = []

    for i, (name, nq, terms, r) in enumerate(val_specs):
        X_val[i] = hamiltonian_to_flat_vector(terms, n_max_qubits, max_hamiltonian_terms)
        mask_val[i] = hamiltonian_to_target_mask(terms, n_max_qubits)
        opt_params, opt_e = run_vqe_optimization(terms, nq, rng_seed=100 + i)
        y_val[i, :nq] = opt_params
        y_val[i, n_max_qubits: n_max_qubits + nq] = opt_params
        E_val[i] = opt_e
        meta_val.append({"molecule": name, "n_qubits": nq, "bond_length": r, "terms": terms})

    # 3. Test Set: Held-Out Out-of-Distribution (BeH2 6-qubit, H4 chain 8-qubit)
    test_specs = [
        ("BeH2", 6, beh2_hamiltonian_sto3g(r), r) for r in [1.0, 1.3, 1.6, 2.0, 2.5]
    ] + [
        ("H4", 8, h4_chain_hamiltonian(r), r) for r in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    ]

    X_test = np.zeros((len(test_specs), d_flat), dtype=np.float32)
    y_test = np.zeros((len(test_specs), n_max_qubits * 2), dtype=np.float32)
    mask_test = np.zeros((len(test_specs), n_pairs), dtype=np.float32)
    E_test = np.zeros(len(test_specs), dtype=np.float32)
    meta_test = []

    for i, (name, nq, terms, r) in enumerate(test_specs):
        X_test[i] = hamiltonian_to_flat_vector(terms, n_max_qubits, max_hamiltonian_terms)
        mask_test[i] = hamiltonian_to_target_mask(terms, n_max_qubits)
        opt_params, opt_e = run_vqe_optimization(terms, nq, rng_seed=200 + i)
        y_test[i, :nq] = opt_params
        y_test[i, n_max_qubits: n_max_qubits + nq] = opt_params
        E_test[i] = opt_e
        meta_test.append({"molecule": name, "n_qubits": nq, "bond_length": r, "terms": terms})

    return {
        "train": {"X": X_train, "y": y_train, "mask": mask_train, "E": E_train, "meta": meta_train},
        "val_interpolation": {"X": X_val, "y": y_val, "mask": mask_val, "E": E_val, "meta": meta_val},
        "test_ood": {"X": X_test, "y": y_test, "mask": mask_test, "E": E_test, "meta": meta_test},
    }

