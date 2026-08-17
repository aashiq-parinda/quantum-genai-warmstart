"""Original Research: GenAI × Quantum Warm-Starting — Full Demonstration."""
import numpy as np
from qwarmstart.data.hamiltonian_encoder import (
    h2_hamiltonian_sto3g, hamiltonian_to_flat_vector, random_hamiltonian
)
from qwarmstart.data.dataset_generator import generate_dataset
from qwarmstart.models.parameter_transformer import ParameterTransformer
from qwarmstart.models.baseline_vqe import run_baseline_vqe, run_vqe_from_init
from qwarmstart.benchmarks.evaluation import run_benchmark, HYPOTHESIS_THRESHOLD_ITER_REDUCTION

print("=" * 75)
print(" 🔬 ORIGINAL RESEARCH: Transformer-Accelerated VQE Warm-Starting")
print("=" * 75)
print()
print("  Hypothesis: A transformer trained on {H → θ_opt} pairs reduces VQE")
print("  convergence iterations by ≥40% vs random initialization.")
print()

# 1. Dataset
print("--- 1. Generating {Hamiltonian → θ_optimal} Training Dataset ---")
dataset = generate_dataset(n_qubits=4, n_samples=80, n_pauli_terms=10,
                           n_params=4, max_hamiltonian_terms=32, rng_seed=42)
print(f"  Samples: {dataset['X'].shape[0]}  Feature dim: {dataset['X'].shape[1]}")
print(f"  Energy range: [{dataset['E'].min():+.4f}, {dataset['E'].max():+.4f}] Hartree")

# 2. Model
print("\n--- 2. Initializing Tiny Transformer Model ---")
model = ParameterTransformer(d_token=17, d_model=32, n_heads=2, n_params=4, seq_len=32)
print(f"  Model parameters: {model.n_parameters():,}")
print(f"  Architecture: {model.d_token}→{model.d_model} (h={model.n_heads}) → {model.n_params}")

# 3. Quick benchmark (untrained) — baseline comparison
print("\n--- 3. Baseline: Random Init VQE on H₂ Hamiltonian ---")
h2_terms = h2_hamiltonian_sto3g()
baseline = run_baseline_vqe(h2_terms, 4, rng_seed=7, n_iters=100)
print(f"  Iterations to converge: {baseline['converged_at']}")
print(f"  Final Energy: {baseline['energy']:+.6f} Hartree")

# 4. Warm-start (untrained transformer — gives random-ish init, shows pipeline)
print("\n--- 4. Warm-Start: Transformer-Predicted Init VQE on H₂ ---")
h2_vec = hamiltonian_to_flat_vector(h2_terms, 4, max_terms=32)
warmstart_params = model.forward(h2_vec)
warmstart = run_vqe_from_init(h2_terms, 4, warmstart_params, n_iters=100)
print(f"  Iterations to converge: {warmstart['converged_at']}")
print(f"  Final Energy: {warmstart['energy']:+.6f} Hartree")
iter_reduction = 1.0 - (warmstart["converged_at"] / max(baseline["converged_at"], 1))
print(f"  Iteration reduction: {iter_reduction:+.1%}")

# 5. Hypothesis evaluation
print("\n--- 5. Hypothesis Evaluation Across Test Hamiltonians ---")
print("  Running benchmark on 10 random Hamiltonians...")
bench_report = run_benchmark(model, n_qubits=4, n_test_hamiltonians=10, max_terms=32)
print(f"  Avg iteration reduction: {bench_report['avg_iter_reduction']:+.1%}")
print(f"  Avg energy difference:   {bench_report['avg_energy_diff_hartree']:.5f} Ha")
if bench_report["hypothesis_supported"]:
    print("  HYPOTHESIS: ✅ SUPPORTED")
else:
    print("  HYPOTHESIS: ⚠️  NOT YET SUPPORTED (untrained model — train for 30+ epochs)")
    print("  This is expected with a randomly-initialized transformer.")
    print("  See docs/PREPRINT_DRAFT.md for full training results.")

print("\n" + "=" * 75)
print(" [✓] GenAI × Quantum Warm-Start Research Pipeline Complete!")
print("=" * 75)
