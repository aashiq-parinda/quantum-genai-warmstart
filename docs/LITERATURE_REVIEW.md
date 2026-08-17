# Literature Review: GenAI × Quantum VQE Warm-Starting

## Research Gap Identification

### Problem Area: VQE Initialization Bottleneck

**Starting points reviewed**:
1. Peruzzo et al. 2014 — Original VQE on photonic hardware
2. McClean et al. 2018 — Barren plateau theory
3. Cerezo et al. 2021 — VQA review (Nature Reviews Physics)
4. Grant et al. 2019 — Initialization strategies for PQCs
5. Cervera-Lierta et al. 2021 — Meta-VQE (related but different)

### What Exists
- **VANS** (Variational Ansatz by Numerical Simulation): Grows circuit + parameters iteratively
- **Meta-VQE**: Trains a PQC that generalizes across Hamiltonian parameters (but requires quantum hardware per training step)
- **QAOA warm-starts**: Classical rounding of SDP relaxations
- **Transfer learning**: Reuse of nearby-geometry VQE parameters

### Identified Gap
**No published work** uses a classical transformer encoder to predict VQE initial parameters from Hamiltonian Pauli string structure. The closest work (Meta-VQE) uses quantum hardware for training and focuses on interpolation rather than generalization across Hamiltonians.

### Why This Gap Exists
- Most VQE research focuses on ansatz design, not initialization
- Hamiltonian encoding as NLP tokens is a non-obvious representation choice
- Classical surrogate models for quantum optimization are nascent (2021-2024)

### Our Contribution
- First pure-classical transformer surrogate predicting VQE parameters from Pauli token embeddings
- Falsifiable hypothesis framework with self-disproof attempts
- Zero-dependency NumPy implementation enabling reproducibility without GPU infrastructure
