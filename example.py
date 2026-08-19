"""Full Research Pipeline: Joint Architecture + Parameter Search for Molecular VQE."""
import numpy as np
from qwarmstart.data.hamiltonian_encoder import (
    h2_hamiltonian_sto3g, lih_hamiltonian_sto3g, beh2_hamiltonian_sto3g, h4_chain_hamiltonian,
)
from qwarmstart.data.dataset_generator import generate_molecular_dataset
from qwarmstart.models.parameter_transformer import ParameterTransformer
from qwarmstart.training.trainer import train_joint_transformer
from qwarmstart.benchmarks.gate_audit import run_full_baseline_gate_audit
from qwarmstart.benchmarks.evaluation import (
    evaluate_joint_vqe_single_system, evaluate_joint_benchmark_suite
)

print("=" * 88)
print(" 🔬 JOINT ARCHITECTURE + PARAMETER SEARCH: FULL RESEARCH EVALUATION")
print("=" * 88)

# -------------------------------------------------------------------------
# PHASE 1: Baseline Gate-Count and Locality Audit
# -------------------------------------------------------------------------
print("\n" + "-" * 88)
print(" [PHASE 1] BASELINE GATE-COUNT & LOCALITY AUDIT")
print("-" * 88)
audit_data = run_full_baseline_gate_audit()
print(f" {'Molecule':<14} | {'Qubits':<6} | {'Total Terms':<11} | {'Local (<=2q)':<12} | {'Fixed CX':<8} | {'Wasted CX':<9} | {'Wasted %':<9}")
print(" " + "-" * 86)
for rec in audit_data["molecules"]:
    print(f" {rec['molecule']:<14} | {rec['n_qubits']:<6} | {rec['total_terms']:<11} | {rec['local_terms_le2']:<12} | {rec['fixed_linear_cx']:<8} | {rec['wasted_linear_cx']:<9} | {rec['pct_wasted_linear']:<8.1f}%")

# -------------------------------------------------------------------------
# PHASE 2 & 3: Dataset Generation & Multi-Objective Model Training
# -------------------------------------------------------------------------
print("\n" + "-" * 88, flush=True)
print(" [PHASE 2 & 3] DATASET GENERATION & JOINT MULTI-OBJECTIVE TRAINING", flush=True)
print("-" * 88, flush=True)
mol_dataset = generate_molecular_dataset(n_max_qubits=8, max_hamiltonian_terms=64, n_random=15, rng_seed=42)
X_train, y_train = mol_dataset["train"]["X"], mol_dataset["train"]["y"]
mask_train = mol_dataset["train"]["mask"]
print(f" Train Samples: {X_train.shape[0]} | Candidate 2-Qubit Pairs: {mask_train.shape[1]}", flush=True)

model = ParameterTransformer(d_token=33, d_model=32, n_heads=2, n_params=16, seq_len=64, n_max_qubits=8, rng_seed=42)
print(f" Initialized ParameterTransformer with Dual Prediction Heads ({model.n_parameters():,} params)", flush=True)

print(" Training with Multi-Objective Loss (MSE + BCE + L1 Sparsity + Connectivity Safeguard)...", flush=True)
train_res = train_joint_transformer(
    model, X_train, y_train, mask_train, n_epochs=25, lr=0.015, lambda_sparse=0.04, lambda_conn=0.02, verbose=True
)
print(f" Final Multi-Objective Loss: {train_res['final_loss']:.5f}", flush=True)

# -------------------------------------------------------------------------
# PHASE 4: Comparative Evaluation Against Fixed-Ansatz Baseline
# -------------------------------------------------------------------------
print("\n" + "-" * 88, flush=True)
print(" [PHASE 4] EVALUATION: JOINT PREDICTED CIRCUITS VS FIXED HEA BASELINE", flush=True)
print("-" * 88, flush=True)

# 1. In-Distribution: H2 (4q, 10 seeds)
h2_terms = h2_hamiltonian_sto3g(0.735)
h2_eval = evaluate_joint_vqe_single_system(model, h2_terms, 4, molecule_name="H2 (0.735 Å)", n_seeds=10)
print(f"\n 1. In-Distribution: H2 (4 qubits, R=0.735 Å, 10 Seeds)", flush=True)
print(f"    - Fixed HEA:       {h2_eval['fixed_cx_count']} CX gates | Depth: {h2_eval['fixed_depth']} | Energy: {h2_eval['energy_mean_fixed']:+.6f} ± {h2_eval['energy_std_fixed']:.6f} Ha | Iters: {h2_eval['iter_mean_fixed']:.1f}", flush=True)
print(f"    - Joint Predicted: {h2_eval['joint_cx_count']} CX gates | Depth: {h2_eval['joint_depth']} | Energy: {h2_eval['energy_mean_joint']:+.6f} ± {h2_eval['energy_std_joint']:.6f} Ha | Iters: {h2_eval['iter_mean_joint']:.1f}", flush=True)
print(f"    - 2-Qubit Gate Reduction: {h2_eval['cx_reduction_pct']:+.1f}% | Energy Diff: {h2_eval['delta_e_mha']:.3f} mHa (Within 5 mHa: {h2_eval['within_target_accuracy']})", flush=True)
print(f"    - Predicted Pairs: {h2_eval['pred_pairs']} (vs Fixed: {h2_eval['fixed_pairs']})", flush=True)
print(f"    - Paired t-test: p(energy) = {h2_eval['p_val_energy']:.4e} | p(iters) = {h2_eval['p_val_iter']:.4e}", flush=True)
print(f"    - Status: {'Strictly Better' if h2_eval['strictly_better'] else ('Pareto Tradeoff' if h2_eval['pareto_tradeoff'] else 'Inferior')}", flush=True)

# 2. Interpolation Benchmark: Unseen Bond Lengths H2 & LiH (5 seeds)
val_eval = evaluate_joint_benchmark_suite(model, mol_dataset["val_interpolation"]["meta"], n_seeds=5)
print(f"\n 2. Interpolation Benchmark (Unseen Bond Lengths H2 & LiH, 5 Seeds):", flush=True)
print(f"    - Average 2-Qubit Gate Reduction: {val_eval['avg_cx_reduction_pct']:+.1f}%", flush=True)
print(f"    - Average Ground-State Energy Error: {val_eval['avg_delta_e_mha']:.3f} mHa", flush=True)
print(f"    - Average Iteration Reduction:       {val_eval['avg_iter_reduction_pct']:+.1f}%", flush=True)
print(f"    - Systems Meeting Target Accuracy (<= 5 mHa): {val_eval['pct_within_target_accuracy']:.0f}%", flush=True)
print(f"    - Systems Supporting Pareto Tradeoff:         {val_eval['pct_pareto_supported']:.0f}%", flush=True)

# 3. Held-Out Zero-Shot OOD Benchmark: BeH2 (6q) and H4 chain (8q) (5 seeds)
ood_eval = evaluate_joint_benchmark_suite(model, mol_dataset["test_ood"]["meta"], n_seeds=5)
print(f"\n 3. Held-Out Zero-Shot OOD Benchmark (BeH2 6q & H4 chain 8q, 5 Seeds):", flush=True)
print(f"    - Average 2-Qubit Gate Reduction: {ood_eval['avg_cx_reduction_pct']:+.1f}%", flush=True)
print(f"    - Average Ground-State Energy Error: {ood_eval['avg_delta_e_mha']:.3f} mHa", flush=True)
print(f"    - Average Iteration Reduction:       {ood_eval['avg_iter_reduction_pct']:+.1f}%", flush=True)
print(f"    - Systems Meeting Target Accuracy (<= 5 mHa): {ood_eval['pct_within_target_accuracy']:.0f}%", flush=True)
print(f"    - Systems Supporting Pareto Tradeoff:         {ood_eval['pct_pareto_supported']:.0f}%", flush=True)

print("\n" + "=" * 88, flush=True)
print(" [✓] Joint Architecture + Parameter Search Evaluation Complete!", flush=True)
print("=" * 88, flush=True)




