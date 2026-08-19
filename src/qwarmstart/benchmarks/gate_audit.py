"""Baseline Gate-Count and Locality Audit Module.

Analyzes molecular Hamiltonian locality vs Fixed Hardware-Efficient Ansatz (HEA)
entangling gate requirements to quantify 'wasted' 2-qubit gates.
"""

from typing import List, Tuple, Dict, Any, Set
import numpy as np


def analyze_hamiltonian_locality(
    pauli_terms: List[Tuple[str, float]],
    n_qubits: int,
) -> Dict[str, Any]:
    """Analyze the weight and locality of Pauli terms in a Hamiltonian.

    Parameters
    ----------
    pauli_terms : List[Tuple[str, float]]
    n_qubits : int

    Returns
    -------
    Dict with:
      - 'total_terms': int
      - 'n_identity': int
      - 'n_1body': int (acts on exactly 1 qubit)
      - 'n_2body': int (acts on exactly 2 qubits)
      - 'n_kbody_gt2': int (acts on >2 qubits)
      - 'local_terms_le2': int (n_1body + n_2body)
      - 'pct_local_le2': float
      - 'interacting_pairs': Set of (i, j) qubit pairs present in 2-body terms
      - 'all_interacting_pairs': Set of (i, j) qubit pairs present in any multi-qubit term
    """
    n_id = 0
    n_1 = 0
    n_2 = 0
    n_gt2 = 0
    interacting_pairs_2body: Set[Tuple[int, int]] = set()
    all_interacting_pairs: Set[Tuple[int, int]] = set()

    for p_str, coeff in pauli_terms:
        # Pad with 'I' if needed
        p = p_str + "I" * max(0, n_qubits - len(p_str))
        active_indices = [i for i, ch in enumerate(p[:n_qubits]) if ch.upper() != "I"]
        weight = len(active_indices)

        if weight == 0:
            n_id += 1
        elif weight == 1:
            n_1 += 1
        elif weight == 2:
            n_2 += 1
            i, j = sorted(active_indices)
            interacting_pairs_2body.add((i, j))
            all_interacting_pairs.add((i, j))
        else:
            n_gt2 += 1
            for idx1 in range(len(active_indices)):
                for idx2 in range(idx1 + 1, len(active_indices)):
                    i, j = sorted([active_indices[idx1], active_indices[idx2]])
                    all_interacting_pairs.add((i, j))

    total_non_id = len(pauli_terms) - n_id
    pct_local = (n_1 + n_2) / max(total_non_id, 1) * 100.0

    return {
        "n_qubits": n_qubits,
        "total_terms": len(pauli_terms),
        "n_identity": n_id,
        "n_1body": n_1,
        "n_2body": n_2,
        "n_kbody_gt2": n_gt2,
        "local_terms_le2": n_1 + n_2,
        "pct_local_le2": pct_local,
        "interacting_pairs_2body": sorted(list(interacting_pairs_2body)),
        "n_interacting_pairs_2body": len(interacting_pairs_2body),
        "total_possible_pairs": n_qubits * (n_qubits - 1) // 2,
    }


def audit_fixed_ansatz_efficiency(
    pauli_terms: List[Tuple[str, float]],
    n_qubits: int,
    n_layers: int = 1,
    topology: str = "linear",  # 'linear' or 'all-to-all'
) -> Dict[str, Any]:
    """Audit 2-qubit entangling gates used by a fixed HEA vs Hamiltonian locality.

    Parameters
    ----------
    pauli_terms : List[Tuple[str, float]]
    n_qubits : int
    n_layers : int — number of entangling layers (default 1)
    topology : str — 'linear' (nearest-neighbor) or 'all-to-all'

    Returns
    -------
    Dict containing gate counts, active pairs, and wasted gate analysis.
    """
    loc = analyze_hamiltonian_locality(pauli_terms, n_qubits)
    h_pairs = set(tuple(p) for p in loc["interacting_pairs_2body"])

    # Define pairs entangled by fixed ansatz per layer
    if topology == "linear":
        ansatz_pairs = [(i, i + 1) for i in range(n_qubits - 1)]
    elif topology == "all-to-all":
        ansatz_pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    else:
        raise ValueError(f"Unknown topology {topology}")

    ansatz_pairs_set = set(ansatz_pairs)
    cx_per_layer = len(ansatz_pairs)
    total_cx = cx_per_layer * n_layers
    single_qubit_rotations = n_qubits * (n_layers + 1)

    # Wasted pairs = pairs entangled by ansatz that have NO 2-body interaction in H
    wasted_pairs = [p for p in ansatz_pairs if p not in h_pairs]
    # Missing pairs = 2-body interacting pairs in H not covered by ansatz
    missing_pairs = [p for p in h_pairs if p not in ansatz_pairs_set]

    wasted_cx_total = len(wasted_pairs) * n_layers
    pct_wasted = (len(wasted_pairs) / max(len(ansatz_pairs), 1)) * 100.0

    return {
        "molecule_n_qubits": n_qubits,
        "n_layers": n_layers,
        "topology": topology,
        "total_cx_gates": total_cx,
        "single_qubit_rotations": single_qubit_rotations,
        "circuit_depth": n_layers * 2 + 1,
        "ansatz_pairs": ansatz_pairs,
        "h_interacting_pairs": sorted(list(h_pairs)),
        "wasted_pairs": wasted_pairs,
        "missing_pairs": missing_pairs,
        "wasted_cx_count": wasted_cx_total,
        "pct_wasted_cx": pct_wasted,
        "hamiltonian_locality": loc,
    }


def run_full_baseline_gate_audit() -> Dict[str, Any]:
    """Run baseline gate audit across all molecular benchmark systems."""
    from qwarmstart.data.hamiltonian_encoder import (
        h2_hamiltonian_sto3g, lih_hamiltonian_sto3g, beh2_hamiltonian_sto3g, h4_chain_hamiltonian
    )

    molecules = [
        ("H2 (4q)", 4, h2_hamiltonian_sto3g(0.735)),
        ("LiH (6q)", 6, lih_hamiltonian_sto3g(1.6)),
        ("BeH2 (6q)", 6, beh2_hamiltonian_sto3g(1.3)),
        ("H4 (8q)", 8, h4_chain_hamiltonian(1.0)),
    ]

    audit_records = []
    for name, nq, terms in molecules:
        audit_linear = audit_fixed_ansatz_efficiency(terms, nq, n_layers=1, topology="linear")
        audit_records.append({
            "molecule": name,
            "n_qubits": nq,
            "total_terms": audit_linear["hamiltonian_locality"]["total_terms"],
            "local_terms_le2": audit_linear["hamiltonian_locality"]["local_terms_le2"],
            "pct_local_le2": audit_linear["hamiltonian_locality"]["pct_local_le2"],
            "h_pairs_count": audit_linear["hamiltonian_locality"]["n_interacting_pairs_2body"],
            "possible_pairs": audit_linear["hamiltonian_locality"]["total_possible_pairs"],
            "fixed_linear_cx": audit_linear["total_cx_gates"],
            "wasted_linear_cx": audit_linear["wasted_cx_count"],
            "pct_wasted_linear": audit_linear["pct_wasted_cx"],
            "linear_wasted_pairs": audit_linear["wasted_pairs"],
            "linear_missing_pairs": audit_linear["missing_pairs"],
        })

    return {"molecules": audit_records}
