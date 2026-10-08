"""Full Research Pipeline: Multi-Seed Training, 4-Baseline Evaluation, & Joint Architecture Search."""
import numpy as np
from qwarmstart.data.hamiltonian_encoder import (
    h2_hamiltonian_sto3g, lih_hamiltonian_sto3g, beh2_hamiltonian_sto3g, h4_chain_hamiltonian,
)
from qwarmstart.data.dataset_generator import generate_molecular_dataset
from qwarmstart.models.parameter_transformer import ParameterTransformer
from qwarmstart.training.trainer import train_joint_transformer, train_multi_seed
from qwarmstart.benchmarks.gate_audit import run_full_baseline_gate_audit
from qwarmstart.benchmarks.evaluation import (
    evaluate_single_hamiltonian_multi_seed,
    evaluate_molecular_suite_multi_seed,
    evaluate_joint_vqe_single_system,
    evaluate_joint_benchmark_suite,
    run_barren_plateau_diagnostic,
)

N_TRAIN_SEEDS = 5
N_EVAL_SEEDS = 5

print("=" * 88)
print(" 🔬 FULL RESEARCH PIPELINE: MULTI-SEED, 4-BASELINE EVALUATION")
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
# PHASE 2: Dataset Generation & Multi-Seed Training (5 Seeds)
# -------------------------------------------------------------------------
print("\n" + "-" * 88, flush=True)
print(f" [PHASE 2] DATASET GENERATION & MULTI-SEED TRAINING ({N_TRAIN_SEEDS} Seeds)", flush=True)
print("-" * 88, flush=True)
mol_dataset = generate_molecular_dataset(n_max_qubits=8, max_hamiltonian_terms=64, n_random=15, rng_seed=42)
X_train, y_train = mol_dataset["train"]["X"], mol_dataset["train"]["y"]
mask_train = mol_dataset["train"]["mask"]
print(f" Train Samples: {X_train.shape[0]} | Candidate 2-Qubit Pairs: {mask_train.shape[1]}", flush=True)

multi_seed_res = train_multi_seed(
    X_train, y_train, mask_train,
    n_seeds=N_TRAIN_SEEDS, n_epochs=20, lr=0.015,
    lambda_sparse=0.04, lambda_conn=0.02,
    d_token=33, d_model=32, n_heads=2, n_params=16,
    seq_len=64, n_max_qubits=8, verbose=True,
)
model = multi_seed_res["best_model"]

print(f"\n Training Loss Across {N_TRAIN_SEEDS} Seeds:", flush=True)
print(f"   Mean ± Std: {multi_seed_res['loss_mean']:.5f} ± {multi_seed_res['loss_std']:.5f}", flush=True)
print(f"   Per-Seed Losses: {['%.5f' % l for l in multi_seed_res['all_final_losses']]}", flush=True)
print(f"   Best Seed: {multi_seed_res['best_seed']} (Loss = {multi_seed_res['best_loss']:.5f})", flush=True)
print(f"   Model Parameters: {model.n_parameters():,}", flush=True)

# -------------------------------------------------------------------------
# PHASE 3: Multi-Molecule 4-Baseline Evaluation (H2, LiH, BeH2, H4)
# -------------------------------------------------------------------------
print("\n" + "-" * 88, flush=True)
print(f" [PHASE 3] 4-BASELINE MULTI-MOLECULE EVALUATION ({N_EVAL_SEEDS} Seeds)", flush=True)
print("-" * 88, flush=True)

eval_systems = [
    ("H2 (In-Dist)", 4, h2_hamiltonian_sto3g(0.735), "H2", "In-Distribution"),
    ("LiH (In-Dist)", 6, lih_hamiltonian_sto3g(1.6), "LiH", "In-Distribution"),
    ("BeH2 (OOD)", 6, beh2_hamiltonian_sto3g(1.3), "BeH2", "Zero-Shot OOD"),
    ("H4 chain (OOD)", 8, h4_chain_hamiltonian(1.0), "H4", "Zero-Shot OOD"),
]

print(f"\n {'System':<18} | {'Regime':<16} | {'Random Energy':<22} | {'HF Energy':<22} | {'PA-HF Energy':<22} | {'Transformer Energy':<22} | {'p(T vs R)':<12} | {'p(T vs HF)':<12}", flush=True)
print(" " + "-" * 160, flush=True)

for name, nq, terms, mol, regime in eval_systems:
    res = evaluate_single_hamiltonian_multi_seed(
        model, terms, nq, molecule_name=mol, n_seeds=N_EVAL_SEEDS,
        max_terms=64, n_max_qubits=8,
    )
    r_e = f"{res['energy_mean_base']:+.4f}±{res['energy_std_base']:.4f}"
    h_e = f"{res['energy_mean_hf']:+.4f}±{res['energy_std_hf']:.4f}"
    p_e = f"{res['energy_mean_pahf']:+.4f}±{res['energy_std_pahf']:.4f}"
    w_e = f"{res['energy_mean_warm']:+.4f}±{res['energy_std_warm']:.4f}"
    p_r = f"{res['p_value_ttest']:.2e}"
    p_h = f"{res['p_value_warm_vs_hf']:.2e}"
    print(f" {name:<18} | {regime:<16} | {r_e:<22} | {h_e:<22} | {p_e:<22} | {w_e:<22} | {p_r:<12} | {p_h:<12}", flush=True)

    # Print iteration summary
    print(f"   {'':18} | {'Iters':<16} | {'Random':<22} | {'HF':<22} | {'PA-HF':<22} | {'Transformer':<22} | {'Beats HF?':<12} | {'Beats PA-HF?':<12}", flush=True)
    r_i = f"{res['iter_mean_base']:.1f}±{res['iter_std_base']:.1f}"
    h_i = f"{res['iter_mean_hf']:.1f}±{res['iter_std_hf']:.1f}"
    p_i = f"{res['iter_mean_pahf']:.1f}±{res['iter_std_pahf']:.1f}"
    w_i = f"{res['iter_mean_warm']:.1f}±{res['iter_std_warm']:.1f}"
    b_hf = "✅ YES" if res['beats_hartree_fock'] else "❌ NO"
    b_pa = "✅ YES" if res['beats_pahf'] else "❌ NO"
    print(f"   {'':18} | {'':16} | {r_i:<22} | {h_i:<22} | {p_i:<22} | {w_i:<22} | {b_hf:<12} | {b_pa:<12}", flush=True)
    print(flush=True)

# -------------------------------------------------------------------------
# PHASE 4: Joint Architecture + Parameter Search
# -------------------------------------------------------------------------
print("-" * 88, flush=True)
print(" [PHASE 4] JOINT PREDICTED CIRCUITS VS FIXED HEA BASELINE", flush=True)
print("-" * 88, flush=True)

# In-Distribution: H2 (4q, 5 seeds)
h2_terms = h2_hamiltonian_sto3g(0.735)
h2_eval = evaluate_joint_vqe_single_system(model, h2_terms, 4, molecule_name="H2 (0.735 Å)", n_seeds=N_EVAL_SEEDS)
print(f"\n 1. In-Distribution: H2 (4 qubits, R=0.735 Å, {N_EVAL_SEEDS} Seeds)", flush=True)
print(f"    - Fixed HEA:       {h2_eval['fixed_cx_count']} CX gates | Energy: {h2_eval['energy_mean_fixed']:+.6f} ± {h2_eval['energy_std_fixed']:.6f} Ha | Iters: {h2_eval['iter_mean_fixed']:.1f}", flush=True)
print(f"    - Joint Predicted: {h2_eval['joint_cx_count']} CX gates | Energy: {h2_eval['energy_mean_joint']:+.6f} ± {h2_eval['energy_std_joint']:.6f} Ha | Iters: {h2_eval['iter_mean_joint']:.1f}", flush=True)
print(f"    - 2-Qubit Gate Reduction: {h2_eval['cx_reduction_pct']:+.1f}% | ΔE: {h2_eval['delta_e_mha']:.3f} mHa", flush=True)
print(f"    - Predicted Pairs: {h2_eval['pred_pairs']} (vs Fixed: {h2_eval['fixed_pairs']})", flush=True)
print(f"    - Status: {'Strictly Better' if h2_eval['strictly_better'] else ('Pareto Tradeoff' if h2_eval['pareto_tradeoff'] else 'Inferior')}", flush=True)

# Interpolation Benchmark
val_eval = evaluate_joint_benchmark_suite(model, mol_dataset["val_interpolation"]["meta"], n_seeds=N_EVAL_SEEDS)
print(f"\n 2. Interpolation (Unseen Bond Lengths, {N_EVAL_SEEDS} Seeds):", flush=True)
print(f"    - Avg 2q Gate Reduction: {val_eval['avg_cx_reduction_pct']:+.1f}% | Avg ΔE: {val_eval['avg_delta_e_mha']:.3f} mHa", flush=True)
print(f"    - Systems ≤ 5 mHa Accuracy: {val_eval['pct_within_target_accuracy']:.0f}% | Pareto: {val_eval['pct_pareto_supported']:.0f}%", flush=True)

# OOD Benchmark
ood_eval = evaluate_joint_benchmark_suite(model, mol_dataset["test_ood"]["meta"], n_seeds=N_EVAL_SEEDS)
print(f"\n 3. Zero-Shot OOD (BeH2 6q & H4 8q, {N_EVAL_SEEDS} Seeds):", flush=True)
print(f"    - Avg 2q Gate Reduction: {ood_eval['avg_cx_reduction_pct']:+.1f}% | Avg ΔE: {ood_eval['avg_delta_e_mha']:.3f} mHa", flush=True)
print(f"    - Systems ≤ 5 mHa Accuracy: {ood_eval['pct_within_target_accuracy']:.0f}% | Pareto: {ood_eval['pct_pareto_supported']:.0f}%", flush=True)

# -------------------------------------------------------------------------
# PHASE 5: Barren Plateau Diagnostic
# -------------------------------------------------------------------------
print("\n" + "-" * 88, flush=True)
print(" [PHASE 5] BARREN PLATEAU DIAGNOSTIC", flush=True)
print("-" * 88, flush=True)
bp_res = run_barren_plateau_diagnostic(model=model, n_samples=50, max_terms=64, n_max_qubits=8)
print(f"\n {'System':<22} | {'Qubits':<6} | {'Var[∇E] Random':<16} | {'Var[∇E] HF':<16} | {'Var[∇E] Transformer':<20} | {'Ratio T/R':<10}")
print(" " + "-" * 100)
for r in bp_res["results"]:
    var_t = r.get("var_transformer", float("nan"))
    ratio = r.get("ratio_trans_vs_random", float("nan"))
    print(f" {r['system']:<22} | {r['n_qubits']:<6} | {r['var_random']:<16.6f} | {r['var_hf']:<16.6f} | {var_t:<20.6f} | {ratio:<10.2f}×")

print("\n" + "=" * 88, flush=True)
print(" [✓] FULL RESEARCH PIPELINE COMPLETE", flush=True)
print("=" * 88, flush=True)
