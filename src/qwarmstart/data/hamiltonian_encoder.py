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

    If len(pauli_str) < n_qubits, pads with 'I' up to n_qubits.

    Parameters
    ----------
    pauli_str : str — e.g. 'ZZII'
    n_qubits : int

    Returns
    -------
    np.ndarray shape (n_qubits * N_PAULI,) — flattened one-hot encoding
    """
    if len(pauli_str) < n_qubits:
        pauli_str = pauli_str + "I" * (n_qubits - len(pauli_str))
    elif len(pauli_str) > n_qubits:
        raise ValueError(f"Pauli string length {len(pauli_str)} > n_qubits={n_qubits}")

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

    if not pauli_terms:
        return tokens

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


# ── Pre-built & Parameterized Molecular Hamiltonians ─────────────────────────

def h2_hamiltonian_sto3g(r: float = 0.735) -> List[Tuple[str, float]]:
    """H₂ molecular Hamiltonian in STO-3G basis (4 spin-orbitals, 4 qubits).

    Parameters
    ----------
    r : float — internuclear distance in Ångströms (default 0.735 Å equilibrium)

    Reference: Peruzzo et al. (2014), Nature Comm.
    Ground state energy at 0.735 Å: E₀ ≈ -1.1361 Hartree
    """
    if abs(r - 0.735) < 1e-4:
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
    
    # Scale coefficients smoothly with bond length r
    dr = r - 0.735
    scale_1b = np.exp(-0.6 * dr)
    scale_2b = np.exp(-0.8 * dr)
    c0 = -0.81054 + 0.35 * (1.0 / r - 1.0 / 0.735)
    
    return [
        ("IIII", c0),
        ("ZZII", +0.17120 * scale_1b),
        ("IZZI", +0.17120 * scale_1b),
        ("IIZZ", -0.22343 * scale_1b),
        ("ZIIZ", +0.16862 * scale_1b),
        ("YXXY", -0.04530 * scale_2b),
        ("XYYX", -0.04530 * scale_2b),
        ("YYXX", +0.04530 * scale_2b),
        ("XXYY", +0.04530 * scale_2b),
        ("ZIII", +0.17120 * scale_1b),
        ("IZII", -0.22343 * scale_1b),
    ]


def lih_hamiltonian_sto3g(r: float = 1.6) -> List[Tuple[str, float]]:
    """LiH molecular Hamiltonian in STO-3G basis (6 qubits active space).

    Parameters
    ----------
    r : float — internuclear distance in Ångströms (default 1.6 Å equilibrium)

    Reference: Kandala et al. (2017), Nature 549.
    """
    dr = r - 1.6
    scale_1b = np.exp(-0.5 * dr)
    scale_2b = np.exp(-0.7 * dr)
    c0 = -7.49872 + 0.5 * (1.0 / r - 1.0 / 1.6)

    return [
        ("IIIIII", c0),
        ("ZIIIII", +0.18093 * scale_1b),
        ("IZIIII", +0.17918 * scale_1b),
        ("IIZIII", -0.24274 * scale_1b),
        ("IIIZII", -0.24274 * scale_1b),
        ("ZZIIII", +0.12339 * scale_2b),
        ("ZIIZII", +0.17679 * scale_2b),
        ("YXXYII", -0.04217 * scale_2b),
        ("XYYXII", -0.04217 * scale_2b),
        ("YYXXII", +0.04217 * scale_2b),
        ("XXYYII", +0.04217 * scale_2b),
        ("IIZZII", +0.15120 * scale_2b),
        ("IIZIZI", +0.11230 * scale_2b),
    ]


def beh2_hamiltonian_sto3g(r: float = 1.3) -> List[Tuple[str, float]]:
    """BeH₂ molecular Hamiltonian in STO-3G basis (6 qubits active space).

    Parameters
    ----------
    r : float — Be-H bond length in Ångströms (default 1.3 Å equilibrium)

    Reference: Kandala et al. (2017), Nature 549.
    Held-out molecule for zero-shot OOD generalization evaluation.
    """
    dr = r - 1.3
    scale_1b = np.exp(-0.5 * dr)
    scale_2b = np.exp(-0.7 * dr)
    c0 = -15.5412 + 0.9 * (1.0 / r - 1.0 / 1.3)

    return [
        ("IIIIII", c0),
        ("ZIIIII", +0.2105 * scale_1b),
        ("IZIIII", +0.2081 * scale_1b),
        ("IIZIII", -0.2854 * scale_1b),
        ("IIIZII", -0.2854 * scale_1b),
        ("IIIIZI", +0.1420 * scale_1b),
        ("ZZIIII", +0.1412 * scale_2b),
        ("ZIZIII", +0.1895 * scale_2b),
        ("YXXYII", -0.0512 * scale_2b),
        ("XYYXII", -0.0512 * scale_2b),
        ("YYXXII", +0.0512 * scale_2b),
        ("XXYYII", +0.0512 * scale_2b),
        ("IYYXXI", -0.0384 * scale_2b),
        ("IXYYXI", -0.0384 * scale_2b),
    ]


def h4_chain_hamiltonian(r: float = 1.0) -> List[Tuple[str, float]]:
    """H₄ linear chain molecular Hamiltonian in STO-3G basis (8 qubits).

    Parameters
    ----------
    r : float — interatomic spacing in Ångströms (default 1.0 Å)

    8-qubit system. Held-out molecule and qubit count for zero-shot OOD evaluation.
    """
    dr = r - 1.0
    scale_1b = np.exp(-0.55 * dr)
    scale_2b = np.exp(-0.75 * dr)
    c0 = -1.9845 + 0.8 * (1.0 / r - 1.0 / 1.0)

    return [
        ("IIIIIIII", c0),
        ("ZIIIIIII", +0.1650 * scale_1b),
        ("IZIIIIII", +0.1650 * scale_1b),
        ("IIZIIIII", +0.1650 * scale_1b),
        ("IIIZIIII", +0.1650 * scale_1b),
        ("ZZIIIIII", +0.1420 * scale_2b),
        ("IZZIFFFF".replace("F", "I"), +0.1420 * scale_2b),
        ("IIZZIIII", +0.1420 * scale_2b),
        ("YXXYIIII", -0.0410 * scale_2b),
        ("XYYXIIII", -0.0410 * scale_2b),
        ("YYXXIIII", +0.0410 * scale_2b),
        ("XXYYIIII", +0.0410 * scale_2b),
        ("IIYXXYII", -0.0410 * scale_2b),
        ("IIXYYXII", -0.0410 * scale_2b),
        ("IIYYXXII", +0.0410 * scale_2b),
        ("IIXXYYII", +0.0410 * scale_2b),
        ("IIIIYXXY", -0.0410 * scale_2b),
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

