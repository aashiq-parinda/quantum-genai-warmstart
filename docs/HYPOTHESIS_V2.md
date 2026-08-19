# Research Hypothesis V2: Joint Architecture and Parameter Search for Molecular VQE

**Date**: August 2026  
**Author**: Ashraf Khan  
**Status**: Experimental Design & Falsifiable Pre-Registration  

---

## 1. Core Hypothesis Statement

> **"A model that jointly predicts which qubit pairs need entangling gates AND their initial parameters, conditioned on Hamiltonian structure, will reach the same target energy accuracy as a fixed hardware-efficient ansatz using significantly fewer 2-qubit gates."**

---

## 2. Motivation & Theoretical Context

### 2.1 The Inefficiency of Fixed Hardware-Efficient Ansätze
Standard Hardware-Efficient Ansätze (HEA) apply a rigid, uniform entangling topology (e.g., linear nearest-neighbor or all-to-all CNOT/CZ chains across all adjacent qubit pairs $(i, i+1)$) independent of the molecular system.

However, fermionic Hamiltonians mapped to qubit operators via Jordan-Wigner transformations exhibit structured locality:
- Many Pauli terms $P_k = \sigma_1^{(k)} \otimes \dots \otimes \sigma_N^{(k)}$ act non-trivially (non-identity) on only 1 or 2 specific qubits.
- Many qubit pairs $(i, j)$ have zero or negligible direct interaction terms in $H$.
- Fixed HEA circuits waste 2-qubit entangling gates on non-interacting qubit pairs, increasing circuit depth, exacerbating decoherence, and inflating NISQ error accumulation (2-qubit gate error rates are typically $10\times$ higher than 1-qubit gates).

### 2.2 Joint Structure and Parameter Generation
By conditioning on the full Hamiltonian token representation $\mathcal{H} = \{(P_k, h_k)\}$, an attention-based encoder can simultaneously extract:
1. **Interaction Graph Topology**: Predicting a binary/probability connection mask $\mathbf{m} \in [0, 1]^{N_{\text{pairs}}}$ indicating which qubit pairs require entangling gates (e.g., CNOT).
2. **Optimal Parameter Warm-Start**: Predicting initial variational rotation angles $\boldsymbol{\theta}_0$ specifically conditioned on that sparse circuit topology.

---

## 3. Falsifiable Quantitative Criteria

To empirically substantiate or falsify this hypothesis, the joint prediction model must satisfy the following thresholds against the fixed HEA baseline across a multi-molecule suite ($H_2$, $\text{LiH}$, $\text{BeH}_2$, $H_4$ chain) over 10 random seeds:

| Metric | Baseline (Fixed HEA) | Target for Joint Prediction | Falsification Threshold |
| :--- | :--- | :--- | :--- |
| **2-Qubit Gate Count ($N_{\text{CX}}$)** | Fixed full/linear topology ($N-1$ per layer) | **$\ge 30\%$ reduction** in $N_{\text{CX}}$ | Reduction $< 15\%$ |
| **Ground-State Energy Accuracy ($\Delta E$)** | Baseline $E_{\text{fixed}}$ | $|E_{\text{joint}} - E_{\text{fixed}}| \le 5\,\text{mHa}$ | Energy error $> 5\,\text{mHa}$ (fails target accuracy) |
| **Convergence Iterations ($N_{\text{iter}}$)** | Baseline iterations | $N_{\text{iter}} \le N_{\text{iter, fixed}}$ | Requires $> 1.25\times$ baseline iterations |
| **Circuit Depth ($D$)** | Full HEA depth | Reduced or equal depth | Depth increases |

---

## 4. Explicit Falsification Conditions (Null Results)

The hypothesis will be deemed **falsified / rejected** if any of the following occur:
1. **Accuracy Collapse**: Enforcing 2-qubit gate sparsity prevents the ansatz from expressing essential electron correlation, leading to ground-state energy deviations $> 5\,\text{mHa}$ from the true or baseline ground state.
2. **Trivial Disconnection**: The sparsity loss collapses the predicted architecture to non-entangled single-qubit rotations ($\mathbf{m} \to \mathbf{0}$), failing to reach the correlated ground state.
3. **No Meaningful Sparsification**: The model predicts dense/near-full connectivity ($\ge 85\%$ of all candidate pairs retained) to preserve accuracy, demonstrating that the learned architecture cannot prune gates without loss.
4. **Pareto Domination by Shallow Fixed Ansätze**: A standard fixed HEA with reduced layer count matches or outperforms the joint-predicted sparse circuit in both gate count and energy.

---

## 5. Experimental Roadmap

- **Phase 1**: Baseline gate-count and locality audit (measuring 2-qubit gates, circuit depth, and Pauli term locality $\le 2$ qubits vs entangling pairs).
- **Phase 2**: Architecture prediction head design and tradeoff analysis (shared encoder vs separate model).
- **Phase 3**: Multi-objective loss formulation (energy-accuracy + parameterized sparsity regularization with collapse safeguards).
- **Phase 4**: Multi-molecule, multi-seed comparative benchmark & Pareto tradeoff analysis.
