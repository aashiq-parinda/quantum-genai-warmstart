# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased — v6] — 2026-09-14

### Changes vs. Zenodo v5 ([10.5281/zenodo.22013110](https://zenodo.org/records/22013110))

#### Added
- **Perturbation-Averaged Hartree-Fock (PA-HF) Baseline** (`baseline_vqe.py`):
  A new classical warm-start baseline that generates K random perturbations around Hartree-Fock parameters, runs short VQE scouts on each, and selects the best candidate as initialization. This provides a stronger classical comparison than raw Hartree-Fock without any ML overhead.

- **Multi-Seed Training** (`trainer.py`):
  Added `train_multi_seed()` function that trains the Transformer across N independent random seeds and reports `loss_mean ± loss_std`. The `train_joint_transformer()` function now accepts a configurable `rng_seed` parameter (previously hardcoded to 42).

- **4-Baseline Evaluation** (`evaluation.py`):
  Extended `evaluate_single_hamiltonian_multi_seed()` to run **four** baselines per molecule:
  Random Init, Hartree-Fock, Perturbation-Averaged HF, and Transformer Warm-Start.
  Added corresponding p-values (`p_value_warm_vs_pahf`) and comparison flags (`beats_pahf`).

- **Expanded End-to-End Pipeline** (`example.py`):
  Restructured to run the complete research pipeline:
  Phase 1 (Gate Audit) → Phase 2 (Multi-Seed Training) → Phase 3 (4-Baseline Multi-Molecule Evaluation) → Phase 4 (Joint Architecture Search) → Phase 5 (Barren Plateau Diagnostic).

- **New Test Cases** (`test_warmstart.py`):
  Added `TestPerturbationAveragedHF`, `TestMultiSeedTraining`, and `TestFourBaselineEvaluation` test classes.

- **CHANGELOG.md**: This file.

#### Changed
- `PREPRINT_DRAFT.md`: Updated Section 2 (Hypothesis), Section 4.2 (now includes PA-HF baseline column), added Section 4.5 (Multi-Seed Training Loss), extended Section 6 (Limitations). Abstract updated to reflect 3 comparison baselines.
- `README.md`: Updated to reflect Version 5 DOI and full research findings.

#### Unchanged
- All existing H₂ results remain unchanged; the re-run produces consistent numbers within reported standard deviations.
- MIT License, repository structure conventions, and existing honest discussion of OOD limitations preserved and extended.
