# Original Research: Generalization & Diagnostic Study of Transformer VQE Warm-Starting

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 18/18 Passed](https://img.shields.io/badge/tests-18%2F18%20passing-brightgreen)](#-testing)
[![DOI: 10.5281/zenodo.21979940](https://zenodo.org/badge/DOI/10.5281/zenodo.21979940.svg)](https://doi.org/10.5281/zenodo.21979940)

This repository contains an expanded, empirical generalization study at the intersection of **Generative AI and Quantum Computing**:

> **Ashraf Khan (2026).** *Transformer-Accelerated Variational Quantum Eigensolvers via Parameter Warm-Starting: A Multi-Molecule Generalization & Diagnostic Study.* [DOI: 10.5281/zenodo.21979940](https://doi.org/10.5281/zenodo.21979940).

---

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/c4596c74-9258-4129-92ae-9b9f0ec162bb" />

---

## 🔬 Research Scope & Multi-Molecule Expansion

We evaluate a **13,156-parameter pure-NumPy Transformer Encoder** $\mathcal{F}_W(H)$ predicting initial parameters $\theta_0$ across 4, 6, and 8 qubit systems ($H_2$, $\text{LiH}$, $\text{BeH}_2$, $H_4$ chain) at varying bond lengths ($R \in [0.5, 3.0]\,\text{\AA}$).

### Unified Token Feature Representation ($N_{\text{max}} = 8$)
$$\text{token}_k = [\text{one-hot}(\sigma_1^{(k)}), \dots, \text{one-hot}(\sigma_8^{(k)}), \bar{h}_k]$$

---

## 📊 Rigorous 4-Phase Benchmark Results (10 Seeds Per Experiment)

### 1. In-Distribution & Interpolation ($H_2$, $\text{LiH}$)
- **Iteration Reduction vs Random Init**: **$+37.8 \pm 4.2\%$**
- **Statistical Significance**: Paired $t$-test **$p = 3.4 \times 10^{-5} < 0.05$** (Statistically Significant).

### 2. Comparison with Classical Hartree-Fock (HF) Baseline
- **Finding**: Classical Hartree-Fock initialization ($\theta_{\text{HF}}$) captures $>75\%$ of iteration savings with **zero ML training cost**.
- On held-out zero-shot OOD molecules ($\text{BeH}_2$ 6q & $H_4$ 8q), **Hartree-Fock outperforms the Transformer**, achieving $4.1\,\text{mHa}$ lower ground-state energy and faster convergence ($p = 0.018$).

### 3. Barren Plateau Diagnostic ($\text{Var}[\partial E / \partial \theta]$)
- On 8-qubit $H_4$ chain, Transformer gradient variance drops towards the random-init barren plateau ($\text{Var} = 0.00492$), whereas Hartree-Fock maintains a strong non-vanishing gradient variance ($\text{Var} = 0.03850$, **$13.7\times$ higher than random**).

---

## ⚠️ Honest Limitations Section

1. **Failure of Zero-Shot OOD Generalization**: The Transformer interpolates well on known potential energy surfaces, but fails to generalize zero-shot to novel molecular topologies ($\text{BeH}_2$) or larger qubit dimensions ($H_4$ 8-qubit chain).
2. **Hartree-Fock Baseline Dominance**: Classical Hartree-Fock parameter initialization requires no neural network training or inference latency, yet consistently outperforms the Transformer on unseen molecules.
3. **Idealized Simulation**: Tested under statevector simulation; real NISQ noise (depolarizing noise, readout errors) and system sizes beyond 8 qubits remain unquantified.

---

## 📄 Research Artifacts

- [`docs/PREPRINT_DRAFT.md`](docs/PREPRINT_DRAFT.md): Updated preprint draft reflecting the 4-phase generalization study and null results.
- `src/qwarmstart/models/parameter_transformer.py`: Pure-NumPy 13K-parameter Transformer encoder.
- `src/qwarmstart/data/hamiltonian_encoder.py`: Multi-molecule Pauli string tokenization ($4, 6, 8$ qubits).
- `src/qwarmstart/data/dataset_generator.py`: Multi-molecule dataset generator (Train, Val Interpolation, Test OOD).
- `src/qwarmstart/benchmarks/evaluation.py`: 10-seed paired statistical significance & barren plateau diagnostic framework.

---

## 💻 Quickstart

```bash
# Clone & install
git clone https://github.com/aashiq-parinda/quantum-genai-warmstart.git
cd quantum-genai-warmstart
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run full research pipeline (Phases 1-4: Dataset gen → Transformer training → 3-Way Benchmark → Diagnostic)
python example.py

# Run test suite
pytest tests/ -v
```

---

## 📜 License

MIT License. Free for academic and commercial research.

