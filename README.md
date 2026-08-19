# Edge-Intelligent-Federated-Learning
Real-Time Anomaly Detection
# uavfl — Federated CNN-LSTM Anomaly Detection for UAV Swarms

Reference **PyTorch** implementation of the full architecture and federated
protocol described in *"Edge-Intelligent and Communication-Efficient
Federated Learning for Real-Time Anomaly Detection in Resource-Constrained
UAV Networks."*

This is the production-architecture counterpart to the lightweight
NumPy surrogate used for the paper's reproducible ablation study
(Section VI of the paper). See [How this relates to the paper](#how-this-relates-to-the-paper)
below for exactly how the two connect.

Every module below implements one specific, numbered equation or algorithm
from the paper, so you can go from a paper section straight to the code
that implements it.

| Paper section | Equation / Algorithm | Module |
|---|---|---|
| III-C | CNN-LSTM architecture, Eq. 1 (loss) | `uavfl/model.py` |
| III-D | Eq. 2 (federated objective) | `uavfl/federated.py` |
| III-E | Eq. 3 (top-k), Eq. 4 (quantization) | `uavfl/compression.py` |
| III-F | Eq. 5 (differential privacy) | `uavfl/privacy.py` |
| III-G | Eq. 6–7 (adaptive aggregation) | `uavfl/aggregation.py` |
| III-H | Eq. 8 (Byzantine filtering) | `uavfl/aggregation.py` |
| III-I | Algorithm 1 (federated round) | `uavfl/federated.py`, `uavfl/server.py` |
| III-I | Algorithm 2 (local training) | `uavfl/client.py` |
| IV | Preprocessing pipeline | `uavfl/preprocessing.py` |
| V | Datasets (synthetic + public) | `uavfl/data.py` |
| VI | Evaluation metrics | `uavfl/metrics.py` |

---

## Installation

```bash
python3 -m venv venv && source venv/bin/activate      # optional but recommended
pip install -r requirements.txt
```

Requires Python ≥ 3.10. CPU-only by default; pass `--device cuda` to
`train.py` if a GPU is available (no code changes needed).

## Quick start

Train on the reproducible synthetic UAV-swarm benchmark (default
configuration matches paper Table IV/V):

```bash
python3 train.py
```

This will:
1. Generate 50 non-IID simulated UAV clients (`uavfl/data.py`), each with
   its own telemetry distribution, anomaly rate, and link quality.
2. Fit z-score normalization once and apply it everywhere (no raw data ever
   leaves a client — see `uavfl/preprocessing.py`).
3. Run federated training for 60 rounds using the full protocol: top-10%
   gradient sparsification + 8-bit quantization + staleness/link-quality
   adaptive aggregation + ε=1.0 differential privacy + Byzantine filtering.
4. Print per-round accuracy/F1/communication cost and a final per-anomaly-
   type breakdown.
5. Save `results/final_model.pt`, `results/history.json`,
   `results/final_metrics.json`.

Expect ~2–5 minutes on CPU for the default 60-round / 50-client
configuration; reduce `--rounds`, `--n_clients`, or `--clients_per_round`
for a faster smoke test (see below).

### Fast smoke test (~10 seconds)

```bash
python3 train.py --rounds 6 --clients_per_round 5 --n_clients 10 \
                  --windows_per_client 40 --local_epochs 2 --eval_every 2
```

Use this to confirm the pipeline runs end-to-end on your machine before
committing to a full run. With this few rounds/epochs the model will not
have converged (expected) — see [Notes on convergence](#notes-on-convergence).

### Reproducing paper-style ablations

Every knob swept in the paper's Section VII (Tables VI–XII) is a CLI flag:

```bash
# Plain FedAvg baseline (no compression, no DP, no adaptive aggregation)
python3 train.py --compression_frac 1.0 --no_dp --out_dir results/fedavg_baseline

# Communication-efficient only (no privacy/robustness extras)
python3 train.py --no_dp --out_dir results/compression_only

# Stress-test Byzantine robustness with defenses OFF
python3 train.py --byzantine_frac 0.2 --robust_filter False --out_dir results/byzantine_unprotected

# Stress-test Byzantine robustness with defenses ON (default)
python3 train.py --byzantine_frac 0.2 --out_dir results/byzantine_protected

# Sweep the privacy budget
for eps in 0.5 1 2 5 10; do
  python3 train.py --dp_epsilon $eps --out_dir results/dp_eps_$eps
done

# Sweep connectivity dropout
for dr in 0.0 0.1 0.2 0.3 0.4; do
  python3 train.py --dropout_rate $dr --out_dir results/dropout_$dr
done

# Scale sweep
for k in 10 25 50; do
  python3 train.py --clients_per_round $k --out_dir results/scale_$k
done
```

Each run writes its own `history.json` (per-round metrics — feed this to
your own plotting script, or adapt the `make_figures.py` from the
paper's companion NumPy package) and `final_metrics.json`.

## Package layout

```
.
├── train.py                 # CLI entry point
├── config.py                 # Default hyperparameters (paper Table V)
├── requirements.txt
├── README.md
└── uavfl/
    ├── __init__.py
    ├── model.py               # CNN-LSTM architecture (Section III-C)
    ├── data.py                 # Synthetic benchmark + public dataset stubs (Section V)
    ├── preprocessing.py        # Windowing, normalization, descriptors (Section IV)
    ├── compression.py          # Top-k sparsification + int8 quantization (Eq. 3-4)
    ├── privacy.py               # Differential privacy (Eq. 5)
    ├── aggregation.py           # FedAvg / adaptive / Byzantine filter (Eq. 6-8)
    ├── client.py                 # UAVClient: local training (Algorithm 2)
    ├── server.py                  # GroundStation: aggregation (Algorithm 1, server side)
    ├── federated.py                # run_federated_training() orchestration loop
    └── metrics.py                   # Accuracy/precision/recall/F1, per-anomaly-type recall
```

Every module has a `if __name__ == "__main__":` block with a runnable,
self-contained example — run any file directly (e.g. `python3
uavfl/compression.py`) to see it work in isolation.

## How this relates to the paper

The paper reports its Section VII numerical results (12 tables, 15
figures) using a **lightweight NumPy statistical-descriptor surrogate**
model, chosen specifically so that every step of the federated protocol
could be exactly audited and re-run quickly and deterministically for a
large ablation grid (compression fraction × quantization × adaptive
aggregation × DP epsilon × dropout rate × Byzantine fraction). That NumPy
code ships separately as `simulate.py` / `run_experiments.py` /
`make_figures.py` in the paper's reproducibility package.

**This repository is the production architecture referenced throughout
the paper as Section III-C** — the actual raw-window CNN-LSTM, in
PyTorch, wired into the identical federated protocol (same equations, same
algorithm). The two implementations share:
- the same four synthetic anomaly mechanisms (`ANOMALY_TYPES` in
  `uavfl/data.py`, matching `simulate.py`),
- the same compression / quantization / DP / adaptive-aggregation /
  Byzantine-filtering math (Eqs. 3–8),
- the same CLI-exposed hyperparameter surface (Table V).

They differ in exactly one respect: this package trains the *full*
CNN-LSTM directly on raw normalized windows, rather than a 140-dimensional
statistical summary. Running the ablations in this repository (see above)
will therefore give you real, freshly-computed numbers for the actual
production architecture, at the cost of longer runtime than the NumPy
surrogate. **Neither codebase's output numbers should be assumed identical
to the other's** — that is expected and is exactly the comparison the
paper's Section IX-C ("Future Directions") calls for as the next
validation step.

## Moving to real datasets

`uavfl/data.py::load_public_dataset_stub()` documents, per dataset, exactly
what a loader needs to return and links to the source for each of the four
public datasets cited in the paper (Table II):

| Dataset | Source |
|---|---|
| ALFA | https://github.com/castacks/alfa-dataset (Keipour, Mousaei & Scherer, *IJRR* 2021) |
| UAV-SEAD | cited in AeroTSBoost, arXiv:2605.25639 |
| UAVCAN labeled dataset | arXiv:2212.09268 |
| Drone-Anomaly | arXiv:2209.13363 |

To switch `train.py` from the synthetic benchmark to real data, replace
the call to `build_synthetic_clients(cfg)` with your own function that
returns the same structure: a list of per-client dicts
`{"id", "X", "y", "link_quality"}` plus a pooled `(X_test, y_test,
atype_test)`. Nothing else in the pipeline needs to change — `uavfl/model.py`,
`uavfl/client.py`, `uavfl/server.py`, and `uavfl/federated.py` are all
dataset-agnostic.

## Notes on convergence

The CNN-LSTM is a larger, more expressive model than the paper's NumPy
surrogate and needs a reasonable number of rounds/local epochs to move
past its random initialization, especially under the ~10% class imbalance
typical of UAV anomaly data. A very short smoke-test run (a handful of
rounds, a couple of local epochs, few clients) may show the model
predicting a single class — this is expected undertraining, not a bug. A
minimal centralized sanity check is provided below and in
`tests/test_pipeline.py` to confirm the architecture and training loop are
correctly wired:

```bash
python3 -m tests.test_pipeline
```

This trains a single (non-federated) model for 15 epochs on pooled
synthetic data and asserts that training loss decreases and F1 becomes
non-zero, isolating the model/optimizer code from the federated-protocol
code for debugging purposes.

## Citing

If you use this code, please cite the accompanying paper: Edge-Intelligent
and Communication-Efficient Federated Learning for Real-Time Anomaly
Detection in Resource-Constrained UAV Networks

## License

Provided as-is for research and educational use.
