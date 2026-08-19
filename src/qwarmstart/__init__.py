"""qwarmstart package — GenAI × Quantum: Transformer-Accelerated VQE."""
from qwarmstart.data.hamiltonian_encoder import encode_hamiltonian, h2_hamiltonian_sto3g
from qwarmstart.data.dataset_generator import generate_dataset, evaluate_vqe_energy
from qwarmstart.models.parameter_transformer import ParameterTransformer
from qwarmstart.models.baseline_vqe import run_baseline_vqe, run_vqe_from_init, run_fixed_hea_vqe
from qwarmstart.training.trainer import train_transformer, train_joint_transformer
from qwarmstart.benchmarks.evaluation import evaluate_joint_vqe_single_system, evaluate_joint_benchmark_suite
from qwarmstart.benchmarks.gate_audit import (
    analyze_hamiltonian_locality,
    audit_fixed_ansatz_efficiency,
    run_full_baseline_gate_audit,
)

__all__ = [
    "encode_hamiltonian", "h2_hamiltonian_sto3g",
    "generate_dataset", "evaluate_vqe_energy",
    "ParameterTransformer",
    "run_baseline_vqe", "run_vqe_from_init", "run_fixed_hea_vqe",
    "train_transformer", "train_joint_transformer",
    "evaluate_joint_vqe_single_system", "evaluate_joint_benchmark_suite",
    "analyze_hamiltonian_locality", "audit_fixed_ansatz_efficiency", "run_full_baseline_gate_audit",
]
