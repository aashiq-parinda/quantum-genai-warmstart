"""Phase 1, 2 & 3: Multi-Molecule Generalization, Statistical Rigor & Hartree-Fock Baseline Study."""
import numpy as np
from qwarmstart.data.hamiltonian_encoder import (
    h2_hamiltonian_sto3g, lih_hamiltonian_sto3g, beh2_hamiltonian_sto3g, h4_chain_hamiltonian,
    hamiltonian_to_flat_vector
)
from qwarmstart.data.dataset_generator import generate_molecular_dataset
from qwarmstart.models.parameter_transformer import ParameterTransformer
from qwarmstart.training.trainer import train_transformer
from qwarmstart.models.baseline_vqe import run_baseline_vqe, run_vqe_from_init, run_hartree_fock_vqe
from qwarmstart.benchmarks.evaluation import (
    evaluate_single_hamiltonian_multi_seed, evaluate_molecular_suite_multi_seed, run_barren_plateau_diagnostic
)

print("=" * 85)
print(" 🔬 FULL STUDY (PHASES 1-4) — GENERALIZATION, STATISTICAL RIGOR & BARREN PLATEAU DIAGNOSTIC")
print("=" * 85)
print()
print("  Comparing 3 Parameter Initialization Strategies Across 4, 6, 8 Qubit Systems:")
print("    1. Random Parameter Init (θ_0 ~ Uniform[0, 2π])")
print("    2. Classical Hartree-Fock Init (θ_HF: Occupied = π, Virtual = 0)")
print("    3. Transformer Warm-Start (θ_Transformer)")
print()

# 1. Generate Dataset
print("--- 1. Generating Multi-Molecule Dataset (4q, 6q, 8q) ---")
mol_dataset = generate_molecular_dataset(n_max_qubits=8, max_hamiltonian_terms=64, rng_seed=42)
X_train, y_train = mol_dataset["train"]["X"], mol_dataset["train"]["y"]
print(f"  Train Samples: {X_train.shape[0]} | Feature Dim: {X_train.shape[1]} (64 terms x 33 features)")
print(f"  Interpolation Validation Samples: {len(mol_dataset['val_interpolation']['meta'])}")
print(f"  Held-Out OOD Test Samples:       {len(mol_dataset['test_ood']['meta'])}")

# 2. Model Initialization (d_token = 8*4+1 = 33, seq_len = 64, n_params = 8)
print("\n--- 2. Initializing ParameterTransformer (N_max=8, d_token=33, seq_len=64) ---")
model = ParameterTransformer(d_token=33, d_model=32, n_heads=2, n_params=8, seq_len=64, rng_seed=42)
print(f"  Total Model Parameters: {model.n_parameters():,}")

# 3. Model Training
print("\n--- 3. Training ParameterTransformer on Multi-Molecule Dataset (20 Epochs) ---")
train_res = train_transformer(model, X_train, y_train, n_epochs=20, lr=0.02, verbose=True)
print(f"  Train Loss: {train_res['loss_history'][0]:.6f} → {train_res['final_loss']:.6f}")

# 4. Multi-Seed Benchmark on H2 4-qubit
print("\n--- 4. H₂ Benchmark (4 qubits, R=0.735 Å, 10 Seeds) ---")
h2_terms = h2_hamiltonian_sto3g(0.735)
h2_ms = evaluate_single_hamiltonian_multi_seed(model, h2_terms, n_qubits=4, molecule_name="H2", n_seeds=10, max_terms=64, n_max_qubits=8)
print(f"  1. Random Baseline Energy:      {h2_ms['energy_mean_base']:+.6f} ± {h2_ms['energy_std_base']:.6f} Ha  | Iters: {h2_ms['iter_mean_base']:.1f} ± {h2_ms['iter_std_base']:.1f}")
print(f"  2. Hartree-Fock Baseline Energy:{h2_ms['energy_mean_hf']:+.6f} ± {h2_ms['energy_std_hf']:.6f} Ha  | Iters: {h2_ms['iter_mean_hf']:.1f} ± {h2_ms['iter_std_hf']:.1f}")
print(f"  3. Transformer Warm-Start Energy:{h2_ms['energy_mean_warm']:+.6f} ± {h2_ms['energy_std_warm']:.6f} Ha  | Iters: {h2_ms['iter_mean_warm']:.1f} ± {h2_ms['iter_std_warm']:.1f}")
print(f"  Paired t-test Transformer vs HF: p = {h2_ms['p_value_warm_vs_hf']:.4e} (Beats HF: {h2_ms['beats_hartree_fock']})")

# 5. Interpolation Benchmark
print("\n--- 5. Interpolation Benchmark (Unseen Bond Lengths H₂ & LiH) ---")
val_ms = evaluate_molecular_suite_multi_seed(model, mol_dataset["val_interpolation"]["meta"], n_seeds=10, max_terms=64, n_max_qubits=8)
print(f"  Avg Iteration Reduction vs Random: {val_ms['avg_iter_reduction_vs_random']:+.1%}")
print(f"  Avg Iteration Reduction vs HF:     {val_ms['avg_iter_reduction_vs_hf']:+.1%}")
print(f"  Pct Systems Beating Hartree-Fock: {val_ms['pct_beats_hartree_fock']:.0f}%")

# 6. Held-Out OOD Benchmark
print("\n--- 6. Zero-Shot OOD Benchmark (BeH₂ 6q & H₄ chain 8q) ---")
ood_ms = evaluate_molecular_suite_multi_seed(model, mol_dataset["test_ood"]["meta"], n_seeds=10, max_terms=64, n_max_qubits=8)
print(f"  Avg Iteration Reduction vs Random: {ood_ms['avg_iter_reduction_vs_random']:+.1%}")
print(f"  Avg Iteration Reduction vs HF:     {ood_ms['avg_iter_reduction_vs_hf']:+.1%}")
print(f"  Pct Systems Beating Hartree-Fock: {ood_ms['pct_beats_hartree_fock']:.0f}%")

# 7. Barren Plateau Diagnostic: Direct Gradient Variance Measurement
print("\n--- 7. Barren Plateau Diagnostic: Gradient Variance Var[∂E/∂θ] vs System Size ---")
bp_diag = run_barren_plateau_diagnostic(model, n_samples=100, max_terms=64, n_max_qubits=8)
print(f"  Samples Per System: {bp_diag['n_samples']}")
print(f"  {'System':<22} | {'Random Var':<12} | {'HF Var':<12} | {'WarmStart Var':<12} | {'Ratio Warm/Random':<18}")
print("  " + "-" * 82)
for r in bp_diag["results"]:
    print(f"  {r['system']:<22} | {r['var_random']:<12.6f} | {r['var_hf']:<12.6f} | {r['var_transformer']:<12.6f} | {r['ratio_trans_vs_random']:<18.2f}x")

print("\n" + "=" * 85)
print(" [✓] Rigorous Generalization & Diagnostic Research Suite Complete!")
print("=" * 85)




