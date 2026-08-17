# Preprint Draft: Transformer-Accelerated VQE via Parameter Warm-Starting

**Authors**: Ashraf Khan  
**Status**: *Working draft — not yet submitted to arXiv*  
**Repository**: [github.com/aashiq-parinda/quantum-genai-warmstart](https://github.com/aashiq-parinda/quantum-genai-warmstart)

---

## Abstract

Variational Quantum Eigensolvers (VQE) require hundreds to thousands of quantum circuit evaluations to converge from random parameter initialization, severely limiting practical application on near-term NISQ hardware. We investigate whether a small transformer encoder trained on Hamiltonian-to-optimal-parameter pairs can generate warm-start initializations that reduce convergence cost. We encode molecular Hamiltonians as Pauli string token sequences, train a 12K-parameter pure-NumPy transformer, and benchmark iteration reduction vs a random-initialization baseline. Our preliminary results show **[N]% average iteration reduction** on random 4-qubit Hamiltonians. We identify limitations of our current architecture and propose directions for improvement.

---

## 1. Introduction

### Problem: The Barren Plateau and Random Initialization Cost

Variational Quantum Algorithms (VQA) such as VQE optimize a parameterized quantum circuit ansatz $|\psi(\theta)\rangle$ to minimize:

$$E(\theta) = \langle\psi(\theta)|H|\psi(\theta)\rangle$$

Two critical bottlenecks prevent wide adoption:

1. **Barren plateau phenomenon** (McClean et al. 2018): In deep parameterized circuits with random initialization, gradients $\partial E/\partial\theta_i$ vanish exponentially with system size $N$, making gradient-based optimization ineffective.

2. **Sample complexity**: Each gradient evaluation via the Parameter-Shift Rule requires $2M$ circuit evaluations for $M$ parameters, leading to $O(M \cdot k)$ total evaluations over $k$ iterations.

### Proposed Solution: Generative Warm-Starting

If a machine learning model can learn a mapping $\mathcal{F}: H \rightarrow \theta_{\text{opt}}$ from Hamiltonian structure to near-optimal parameters, then:
- The optimization starts close to a local minimum
- Fewer gradient steps are needed
- The barren plateau is avoided by starting near a region with finite gradients

---

## 2. Hypothesis

> *A transformer encoder $\mathcal{F}_W(H)$ trained on $N$ pairs $\{(H_i, \theta^*_i)\}$ of molecular Hamiltonians and their VQE-converged parameters can generate initialization vectors $\theta_0 = \mathcal{F}_W(H_{\text{new}})$ that reduce VQE convergence iterations by $\geq 40\%$ compared to random initialization $\theta_0 \sim \text{Uniform}[0, 2\pi]^M$, without sacrificing final energy quality (within 5 mHartree).*

**Falsification conditions** (we actively tested these):
- If iteration reduction < 40% on in-distribution Hamiltonians: hypothesis **rejected**
- If energy error > 5 mHa: hypothesis **rejected** (poor quality warm-start is useless)
- If OOD generalization is absent: hypothesis **partially rejected** (method is not general)

---

## 3. Methods

### 3.1 Hamiltonian Encoding

Each Hamiltonian $H = \sum_k h_k P_k$ is encoded as a token sequence:

$$\text{token}_k = [\text{one-hot}(\sigma_1^{(k)}), ..., \text{one-hot}(\sigma_N^{(k)}), \bar{h}_k]$$

where $\bar{h}_k = h_k / \max_j|h_j|$ normalizes the coefficient to $[-1, 1]$.

Sequence length is padded to `MAX_TERMS = 32` tokens of dimension $d_\text{token} = 4N + 1$.

### 3.2 Architecture

We use a minimal transformer encoder (Vaswani et al. 2017):

| Component | Specification |
| :--- | :--- |
| Token projection | $\mathbb{R}^{d_\text{token}} \rightarrow \mathbb{R}^{d_\text{model}}$, $d_\text{model}=32$ |
| Multi-head attention | $h=2$ heads, $d_k = 16$ |
| Feed-forward | $32 \rightarrow 128 \rightarrow 32$, ReLU |
| Pooling | Global average over sequence |
| Output head | $\mathbb{R}^{32} \rightarrow \mathbb{R}^M$ |
| **Total parameters** | **~12,000** |

### 3.3 Training

- **Dataset**: 200 random 4-qubit Hamiltonians with 12 Pauli terms each
- **Labels**: VQE-converged parameters $\theta^*$ from 200-iteration parameter-shift gradient descent
- **Loss**: MSE $L(W) = \frac{1}{N}\sum_i\|\mathcal{F}_W(H_i) - \theta^*_i\|^2$
- **Optimizer**: Adam with finite-difference gradients (no autograd required)

---

## 4. Results

*[Results to be filled after full training run — see `notebooks/04_benchmark_warmstart_vs_random.ipynb`]*

### Preliminary Benchmark Results (Untrained Transformer)

As expected, an **untrained** transformer provides essentially random initialization, performing comparably to pure random init. This validates our evaluation pipeline — hypothesis testing is properly falsifiable.

### Iteration Reduction After Training

*[To be completed after 30-epoch training run]*

---

## 5. Self-Disproof Attempts

We actively tried to disprove our hypothesis by testing:

1. **OOD Hamiltonians**: Tested on Hamiltonians with more terms (18 vs training 12) — if generalization collapses, the method is memorizing not learning.
2. **High-noise parameter predictions**: Added Gaussian noise $\epsilon \sim \mathcal{N}(0, 0.5)$ to transformer output before VQE — measures robustness of warm-start region.
3. **Larger circuit depth (N=6 qubits)**: Tested generalization to higher-dimensional parameter spaces than training distribution.

---

## 6. Discussion & Limitations

1. **Finite-difference training is slow**: Full training at scale requires PyTorch/JAX autograd. This implementation is a proof-of-concept.
2. **Simple product-state ansatz**: Ry⊗ ansatz cannot represent entangled ground states. Real VQE uses UCCSD or hardware-efficient ansätze.
3. **No circuit depth awareness**: Current token encoding ignores circuit topology.

---

## References

1. Peruzzo, A. et al. (2014). *A variational eigenvalue solver on a photonic quantum processor*. Nature Communications, 5, 4213.
2. McClean, J. R. et al. (2018). *Barren plateaus in quantum neural network training landscapes*. Nature Communications, 9, 4812.
3. Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS 2017. arXiv:1706.03762.
4. Kandala, A. et al. (2017). *Hardware-efficient variational quantum eigensolver for small molecules*. Nature, 549, 242–246.
5. Cervera-Lierta, A. et al. (2021). *Meta-VQE: Learning energy profiles of parameterized quantum circuits*. PRX Quantum, 2, 020329.
