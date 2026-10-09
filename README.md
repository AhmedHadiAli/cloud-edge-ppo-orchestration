# Cloud-Edge PPO Reviewer-Revision Reproducibility Package (v2.0)

This package reproduces the reviewer-driven revision of **Design and Evaluation of Adaptive PPO-Based Resource Orchestration for Latency-Sensitive AI Inference in Cloud-Edge Computing**.

## What changed after peer review

The experimental design was rebuilt rather than cosmetically edited. The revised artifact uses one unified PPO policy across all operating conditions, 45 s episodes that cover the complete burst/recovery and four-phase network-fluctuation cycles, re-provisioned device/edge/cloud capacities, serialization-only radio-energy accounting, a binding battery-constrained scenario, and seed-controlled per-task network jitter. The unsupported reduced-precision compute speed-up from the original submission was removed.

The comparator set now includes local-only, edge-only, cloud-only, a non-degenerate static rule, a transparent one-step model-based greedy policy, a validation-selected exhaustive constant-action baseline, and DQN trained with the same mixed-scenario interaction budget. PPO is trained from three independent seeds and DQN from three independent seeds. Final evaluation uses 20 matched task/network seeds. Deterministic and stochastic PPO are both reported.

## Scope and interpretation

The main study remains an **event-based analytical simulation**. Reported device-side energy, DNN retention, and resource capacities are modeled quantities rather than physical power-meter or validation-set measurements. The included `hardware_microbenchmark.py` performs a separate real PyTorch CPU forward-pass timing check for ResNet-50 and MobileNetV2. It is a timing sanity check only; it is not used as a surrogate for an edge/cloud testbed and does not measure energy or classification accuracy.

## Repository contents

- `code/revised_framework.py` – complete simulator, PPO, DQN, baselines, statistics, sensitivity analysis, diagnostics, and figure generation.
- `code/CloudEdge_PPO_ReviewerRevision.ipynb` – small notebook wrapper for running the framework and inspecting outputs.
- `code/hardware_microbenchmark.py` – real-DNN single-thread CPU timing sanity check.
- `config/experiment_config.json` – full model, environment, action, reward, seed, and training configuration.
- `checkpoints/` – three retained PPO and three retained DQN checkpoints.
- `results/results_raw_all_replicates.csv` – per-scenario/per-seed results before aggregation across training replicates.
- `results/results_paired_aggregated.csv` – paired 20-seed evaluation table used for manuscript-level inference.
- `results/results_summary.csv` – mean and sample-SD table.
- `results/paired_statistics_global_holm.csv` – all PPO-vs-comparator Wilcoxon tests in one global Holm family, with effect sizes and bootstrap confidence intervals.
- `results/state_dependence_probe.csv` – 20,000-state actor probes for each PPO training seed.
- `results/scenario_coverage.csv` – burst/recovery and all four fluctuation-phase occupancy counts.
- `results/sensitivity_raw.csv`, `results/sensitivity_summary.csv` – arrival-rate, bandwidth, and capacity sensitivity results.
- `results/ppo_training_history.csv`, `results/dqn_training_history.csv` – learning histories.
- `results/hardware_microbenchmark.csv` – measured CPU timing sanity check.
- `figures/` – manuscript Figures 1–6 in both SVG and high-resolution PNG.
- `verify_repository.py` – fast internal consistency verification.
- `submission/` – clean revised manuscript, red-marked revised manuscript, and point-by-point response in Word/PDF.

## Installation

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Re-run the complete simulation experiment

```bash
python code/revised_framework.py
```

The complete command retrains the three PPO replicas and three DQN replicas, evaluates 20 matched seeds, regenerates statistical and sensitivity outputs, and recreates Figures 1–6.

## Re-run the real-DNN timing sanity check

```bash
python code/hardware_microbenchmark.py
```

## Verify the archived outputs

```bash
python verify_repository.py
```

## Main revised protocol

- Source devices / edge servers / cloud nodes: **4 / 3 / 1**
- Aggregate device / edge / cloud capacity: **104 / 144 / 120 GFLOPS**
- Episode horizon: **45 s**
- Nominal arrival rate: **16 tasks/s**
- Burst/recovery schedule: **2.5x for the first 10 s of each 30 s cycle, then 0.75x**
- Network fluctuation: **four 10 s phases; all phases are traversed within each evaluation episode**
- Unified PPO training seeds: **42, 142, 242**
- Unified DQN training seeds: **123, 223, 323**
- Final matched evaluation seeds: **1000–1019**
- Constant-baseline selection seeds: **7000–7004** (disjoint from final evaluation)
- Sensitivity seeds: **1100–1104**

## Verification status

`python verify_repository.py` passes **107/107** archived-output consistency and protocol checks for this packaged revision.

## Archival note

The package is versioned as **v2.0 reviewer revision** and is ready for GitHub Release / Zenodo deposit. A Zenodo DOI must be created by the repository owner after upload; no DOI has been fabricated in the manuscript or this package.
