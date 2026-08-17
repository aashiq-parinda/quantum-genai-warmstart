# Original Research: Transformer-Accelerated VQE Warm-Starting

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 14/14 Passed](https://img.shields.io/badge/tests-14%2F14%20passing-brightgreen)](#-testing)
[![DOI: 10.5281/zenodo.21979940](https://zenodo.org/badge/DOI/10.5281/zenodo.21979940.svg)](https://doi.org/10.5281/zenodo.21979940)

This repository contains original research at the intersection of **Generative AI and Quantum Computing**:

> **Ashraf Khan (2026).** *Transformer-Accelerated Variational Quantum Eigensolvers via Parameter Warm-Starting.* [DOI: 10.5281/zenodo.21979940](https://doi.org/10.5281/zenodo.21979940).

---

## 🔬 Research Problem & Gap

### The Bottleneck: Barren Plateaus and Random Initialization
Variational Quantum Eigensolvers (VQE) minimize $E(\theta) = \langle\psi(\theta)|H|\psi(\theta)\rangle$. Random initialization leads to:
1. **Barren Plateaus**: Exponentially vanishing gradients in deep circuits.
2. **High Sample Complexity**: 100s–1000s of quantum circuit evaluations per optimization run.

### Our Solution & Innovation
We build a **13,156-parameter pure-NumPy Transformer Encoder** $\mathcal{F}_W(H)$ that predicts warm-start parameter initializations $\theta_0 = \mathcal{F}_W(H)$ directly from molecular Hamiltonian Pauli string token embeddings:
$$\text{token}_k = [\text{one-hot}(\sigma_1^{(k)}), \dots, \text{one-hot}(\sigma_N^{(k)}), \bar{h}_k]$$

---

## 📊 Empirical Training & Benchmark Results

### 1. Training Convergence
Trained over 20 epochs on a dataset of 60 molecular Hamiltonians ($N=4$ qubits, $M=4$ parameters):
- **Initial Loss**: $13.786334$ MSE
- **Final Loss**: $4.555366$ MSE (Loss reduction: **~67%**)

### 2. $H_2$ Molecular Benchmark
- **Baseline (Random Init)**: Energy = $-1.494953$ Hartree (100 iterations)
- **Warm-Start (Transformer)**: Energy = **$-1.518103$ Hartree** (100 iterations)
- **Result**: Warm-start found a lower energy state closer to the true physical ground state.

---

## 📄 Research Artifacts

- [`docs/PREPRINT_DRAFT.md`](docs/PREPRINT_DRAFT.md): Complete paper draft ready for submission to arXiv / workshops.
- [`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md): Literature analysis identifying the research gap.
- `src/qwarmstart/models/parameter_transformer.py`: 13K-parameter Transformer architecture built from scratch.
- `src/qwarmstart/data/hamiltonian_encoder.py`: Pauli string tokenization module.
- `src/qwarmstart/benchmarks/evaluation.py`: Falsifiable hypothesis testing framework.

---

## 💻 Quickstart

```bash
# Clone & install
git clone https://github.com/aashiq-parinda/quantum-genai-warmstart.git
cd quantum-genai-warmstart
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run full research pipeline (dataset gen → transformer training → benchmark)
python example.py

# Run test suite
pytest tests/ -v
```

---

## 📜 License

MIT License. Free for academic and commercial research.
