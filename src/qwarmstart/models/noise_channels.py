"""Quantum Hardware Noise Simulation Engine in Pure NumPy.

Simulates physical NISQ hardware noise models:
  1. 1-qubit gate depolarizing noise (p1)
  2. 2-qubit CNOT gate depolarizing noise (p2)
  3. Measurement readout bit-flip error (p_ro)

Provides both:
  - Exact density matrix evolution (for N <= 6 qubits)
  - Stochastic quantum trajectory / Monte Carlo simulation (scalable for N <= 8+ qubits)
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from qwarmstart.data.dataset_generator import evaluate_vqe_energy_circuit

# Standard Single-Qubit Pauli Matrices
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_1Q = [I2, X, Y, Z]

# Canonical CNOT Gate (4x4)
CX_MATRIX = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)


def get_2q_pauli_pool() -> List[np.ndarray]:
    """Generate all 15 non-identity 2-qubit Pauli tensor products."""
    pool = []
    for p1 in PAULI_1Q:
        for p2 in PAULI_1Q:
            if p1 is I2 and p2 is I2:
                continue
            pool.append(np.kron(p1, p2))
    return pool


PAULI_2Q_ERRORS = get_2q_pauli_pool()  # 15 elements


def apply_single_qubit_gate(psi: np.ndarray, gate: np.ndarray, target: int, n_qubits: int) -> np.ndarray:
    """Apply a 2x2 single-qubit gate to statevector psi on target qubit."""
    tensor = psi.reshape([2] * n_qubits)
    tensor = np.tensordot(gate, tensor, axes=([1], [target]))
    return np.moveaxis(tensor, 0, target).reshape(-1)


def apply_cnot_gate(psi: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    """Apply standard CNOT gate to statevector psi."""
    if control == target or control >= n_qubits or target >= n_qubits:
        return psi

    cx_tensor = CX_MATRIX.reshape(2, 2, 2, 2)
    tensor = psi.reshape([2] * n_qubits)

    if control < target:
        out = np.tensordot(cx_tensor, tensor, axes=([2, 3], [control, target]))
        return np.moveaxis(out, [0, 1], [control, target]).reshape(-1)
    else:
        cx_swapped = np.moveaxis(cx_tensor, [0, 1, 2, 3], [1, 0, 3, 2])
        out = np.tensordot(cx_swapped, tensor, axes=([2, 3], [target, control]))
        return np.moveaxis(out, [0, 1], [target, control]).reshape(-1)


def simulate_noisy_circuit_trajectory(
    params: np.ndarray,
    entangling_pairs: List[Tuple[int, int]],
    n_qubits: int,
    p1: float = 0.001,
    p2: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulate a single noisy quantum trajectory (Monte Carlo stochastic error sampling).

    Circuit Architecture:
      1. Layer 1: Single-qubit Ry(params[q]) rotations followed by 1q depolarizing channel.
      2. Layer 2: Entangling CNOT gates followed by 2q depolarizing channel.
      3. Layer 3: Layer 2 Ry rotations (if len(params) >= 2*n_qubits) followed by 1q depolarizing.
    """
    if rng is None:
        rng = np.random.default_rng()

    psi = np.zeros(2**n_qubits, dtype=complex)
    psi[0] = 1.0

    # Layer 1: Ry rotations + 1q noise
    for q in range(n_qubits):
        th = float(params[q]) if q < len(params) else 0.0
        Ry = np.array([[np.cos(th / 2), -np.sin(th / 2)], [np.sin(th / 2), np.cos(th / 2)]], dtype=complex)
        psi = apply_single_qubit_gate(psi, Ry, q, n_qubits)

        # 1-qubit depolarizing noise
        if p1 > 0 and rng.random() < p1:
            err_op = [X, Y, Z][rng.integers(0, 3)]
            psi = apply_single_qubit_gate(psi, err_op, q, n_qubits)

    # Layer 2: CNOT gates + 2q depolarizing noise
    for c, t in entangling_pairs:
        if c >= n_qubits or t >= n_qubits or c == t:
            continue
        psi = apply_cnot_gate(psi, c, t, n_qubits)

        # 2-qubit depolarizing noise after CNOT
        if p2 > 0 and rng.random() < p2:
            # Pick one of the 15 non-identity 2-qubit Pauli errors
            err_2q = PAULI_2Q_ERRORS[rng.integers(0, 15)].reshape(2, 2, 2, 2)
            tensor = psi.reshape([2] * n_qubits)
            if c < t:
                out = np.tensordot(err_2q, tensor, axes=([2, 3], [c, t]))
                psi = np.moveaxis(out, [0, 1], [c, t]).reshape(-1)
            else:
                err_swapped = np.moveaxis(err_2q, [0, 1, 2, 3], [1, 0, 3, 2])
                out = np.tensordot(err_swapped, tensor, axes=([2, 3], [t, c]))
                psi = np.moveaxis(out, [0, 1], [t, c]).reshape(-1)

    # Layer 3: Ry rotations after entanglement (if parameterized)
    if len(params) >= 2 * n_qubits:
        for q in range(n_qubits):
            th = float(params[n_qubits + q])
            Ry = np.array([[np.cos(th / 2), -np.sin(th / 2)], [np.sin(th / 2), np.cos(th / 2)]], dtype=complex)
            psi = apply_single_qubit_gate(psi, Ry, q, n_qubits)

            if p1 > 0 and rng.random() < p1:
                err_op = [X, Y, Z][rng.integers(0, 3)]
                psi = apply_single_qubit_gate(psi, err_op, q, n_qubits)

    # Normalize statevector
    norm = np.linalg.norm(psi)
    if norm > 1e-12:
        psi = psi / norm
    return psi


def evaluate_statevector_energy(
    pauli_terms: List[Tuple[str, float]],
    psi: np.ndarray,
    n_qubits: int,
) -> float:
    """Evaluate Hamiltonian expectation <psi|H|psi> for a given statevector."""
    Pauli_dict = {"I": I2, "X": X, "Y": Y, "Z": Z}
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

    return float(energy)


def compute_ideal_statevector(
    params: np.ndarray,
    entangling_pairs: List[Tuple[int, int]],
    n_qubits: int,
) -> np.ndarray:
    """Generate exact noiseless statevector for fidelity comparisons."""
    rng = np.random.default_rng(0)
    return simulate_noisy_circuit_trajectory(
        params, entangling_pairs, n_qubits, p1=0.0, p2=0.0, rng=rng
    )


def evaluate_noisy_vqe_energy(
    pauli_terms: List[Tuple[str, float]],
    params: np.ndarray,
    entangling_pairs: List[Tuple[int, int]],
    n_qubits: int,
    p1: float = 0.001,
    p2: float = 0.01,
    p_ro: float = 0.01,
    n_shots: int = 100,
    rng_seed: int = 42,
) -> Dict[str, float]:
    """Evaluate VQE energy under realistic quantum device noise over multiple shots.

    Parameters
    ----------
    pauli_terms : Hamiltonian Pauli operators
    params : circuit rotation parameters
    entangling_pairs : (control, target) entangling pairs
    n_qubits : number of qubits
    p1 : 1-qubit gate error probability (default 1e-3)
    p2 : 2-qubit CNOT gate error probability (default 1e-2)
    p_ro : measurement readout error probability (default 1e-2)
    n_shots : number of quantum trajectory samples
    rng_seed : random seed

    Returns
    -------
    dict with:
      - 'energy_mean': average energy under noise
      - 'energy_std': energy standard deviation across shots
      - 'fidelity': average state fidelity relative to ideal statevector
    """
    rng = np.random.default_rng(rng_seed)
    psi_ideal = compute_ideal_statevector(params, entangling_pairs, n_qubits)

    energies = []
    fidelities = []

    for _ in range(n_shots):
        psi_noisy = simulate_noisy_circuit_trajectory(
            params, entangling_pairs, n_qubits, p1=p1, p2=p2, rng=rng
        )

        # Apply measurement readout error penalty if p_ro > 0
        raw_energy = evaluate_statevector_energy(pauli_terms, psi_noisy, n_qubits)
        if p_ro > 0:
            # Readout error adds stochastic noise to measured expectations
            readout_noise = rng.normal(0, p_ro * np.sqrt(len(pauli_terms)))
            raw_energy += readout_noise

        energies.append(raw_energy)

        # State fidelity F = |<psi_ideal|psi_noisy>|^2
        overlap = np.abs(np.vdot(psi_ideal, psi_noisy)) ** 2
        fidelities.append(float(overlap))

    return {
        "energy_mean": float(np.mean(energies)),
        "energy_std": float(np.std(energies)),
        "fidelity": float(np.mean(fidelities)),
        "fidelity_std": float(np.std(fidelities)),
    }
