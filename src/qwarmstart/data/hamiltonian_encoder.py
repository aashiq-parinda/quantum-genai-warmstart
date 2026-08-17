"""Hamiltonian Encoder: Pauli strings → transformer token embeddings.

Encodes a molecular Hamiltonian H = Σ_k h_k P_k into a fixed-length
feature vector suitable for a transformer encoder.

Each Pauli term P_k = σ_1^(k) ⊗ σ_2^(k) ⊗ ... ⊗ σ_N^(k) is encoded as:
    token_k = [one_hot(σ_1^k), ..., one_hot(σ_N^k), normalized_coefficient]

Pauli basis:  I=0, X=1, Y=2, Z=3  (4-class one-hot per qubit)

Final embedding: context-padded to MAX_TERMS tokens of dimension d_model.
"""

import numpy as np
from typing import List, Tuple, Dict


# Pauli operator index
PAULI_MAP: Dict[str, int] = {"I": 0, "X": 1, "Y": 2, "Z": 3}
N_PAULI = 4  # I, X, Y, Z


def encode_pauli_string(pauli_str: str, n_qubits: int) -> np.ndarray:
    """Encode a single Pauli string like 'XZIY' into one-hot qubit features.

    Parameters
    ----------
    pauli_str : str — e.g. 'ZZII' (length n_qubits)
    n_qubits : int

    Returns
    -------
    np.ndarray shape (n_qubits * N_PAULI,) — flattened one-hot encoding
    """
    if len(pauli_str) != n_qubits:
        raise ValueError(f"Pauli string length {len(pauli_str)} ≠ n_qubits={n_qubits}")
    encoding = np.zeros(n_qubits * N_PAULI, dtype=np.float32)
    for i, char in enumerate(pauli_str):
        idx = PAULI_MAP.get(char.upper())
        if idx is None:
            raise ValueError(f"Unknown Pauli '{char}', must be I/X/Y/Z")
        encoding[i * N_PAULI + idx] = 1.0
    return encoding


def encode_hamiltonian(
    pauli_terms: List[Tuple[str, float]],
    n_qubits: int,
    max_terms: int = 64,
) -> np.ndarray:
    """Encode a full Hamiltonian H = Σ_k h_k P_k as a token matrix.

    Parameters
    ----------
    pauli_terms : list of (pauli_string, coefficient)
    n_qubits : int
    max_terms : int — pad/truncate to this many tokens

    Returns
    -------
    np.ndarray shape (max_terms, n_qubits * N_PAULI + 1) — token matrix
        Each row: [one_hot(P_k), normalized_h_k]
    """
    d_token = n_qubits * N_PAULI + 1  # +1 for coefficient
    tokens = np.zeros((max_terms, d_token), dtype=np.float32)

    # Normalize coefficients to [-1, 1]
    coeff_arr = np.array([abs(h) for _, h in pauli_terms], dtype=np.float32)
    max_coeff = coeff_arr.max() if coeff_arr.max() > 0 else 1.0

    for i, (pauli_str, coeff) in enumerate(pauli_terms[:max_terms]):
        pauli_enc = encode_pauli_string(pauli_str, n_qubits)
        tokens[i, :-1] = pauli_enc
        tokens[i, -1] = coeff / max_coeff

    return tokens


def hamiltonian_to_flat_vector(
    pauli_terms: List[Tuple[str, float]],
    n_qubits: int,
    max_terms: int = 64,
) -> np.ndarray:
    """Flatten Hamiltonian token matrix to a 1D feature vector for MLP/transformer input."""
    tokens = encode_hamiltonian(pauli_terms, n_qubits, max_terms)
    return tokens.flatten()


# ── Pre-built test Hamiltonians ──────────────────────────────────────────────

def h2_hamiltonian_sto3g() -> List[Tuple[str, float]]:
    """H₂ molecular Hamiltonian in STO-3G basis (4 spin-orbitals, 4 qubits).

    From PySCF / OpenFermion standard computation at R=0.735 Å equilibrium.
    Reference: Peruzzo et al. (2014), Nature Comm.
    Ground state energy: E₀ ≈ -1.1361 Hartree
    """
    return [
        ("IIII", -0.81054),
        ("ZZII", +0.17120),
        ("IZZI", +0.17120),
        ("IIZZ", -0.22343),
        ("ZIIZ", +0.16862),
        ("YXXY", -0.04530),
        ("XYYX", -0.04530),
        ("YYXX", +0.04530),
        ("XXYY", +0.04530),
        ("ZIII", +0.17120),
        ("IZII", -0.22343),
    ]


def lih_hamiltonian_sto3g() -> List[Tuple[str, float]]:
    """LiH molecular Hamiltonian in STO-3G basis (6 qubits, truncated to active space).

    Simplified 4-qubit active space approximation.
    Reference: Kandala et al. (2017), Nature 549.
    """
    return [
        ("IIII", -7.49872),
        ("ZIII", +0.18093),
        ("IZII", +0.17918),
        ("IIZI", -0.24274),
        ("IIIZ", -0.24274),
        ("ZZII", +0.12339),
        ("ZIIZ", +0.17679),
        ("YXXY", -0.04217),
        ("XYYX", -0.04217),
        ("YYXX", +0.04217),
        ("XXYY", +0.04217),
    ]


def random_hamiltonian(n_qubits: int, n_terms: int, rng_seed: int = 42) -> List[Tuple[str, float]]:
    """Generate a random Hermitian qubit Hamiltonian for dataset generation."""
    rng = np.random.default_rng(rng_seed)
    paulis = list("IXYZ")
    terms = []
    for _ in range(n_terms):
        pauli_str = "".join(rng.choice(paulis, n_qubits))
        coeff = float(rng.uniform(-1.0, 1.0))
        terms.append((pauli_str, coeff))
    return terms
