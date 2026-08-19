"""
config.py — Default hyperparameters, matching Table V of the paper.
Override any of these from the command line via train.py --help.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # Data / benchmark (Table IV)
    n_clients: int = 50
    windows_per_client: int = 200
    window: int = 50
    n_features: int = 28
    non_iid: str = "severe"          # "iid" | "mild" | "severe"
    data_seed: int = 0

    # Model (Section III-C)
    conv1_filters: int = 32
    conv2_filters: int = 64
    lstm1_units: int = 64
    lstm2_units: int = 32
    dense_units: int = 16
    dropout: float = 0.3

    # Federated protocol (Table V)
    rounds: int = 60
    clients_per_round: int = 20
    local_epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    compression_frac: float = 0.10
    quantize: bool = True
    adaptive_aggregation: bool = True
    dp_epsilon: float = 1.0          # set to None (via CLI --no_dp) to disable
    dp_clip: float = 1.0
    dropout_rate: float = 0.0        # connectivity dropout, distinct from model `dropout`
    byzantine_frac: float = 0.0
    robust_filter: bool = True

    # Runtime
    device: str = "cpu"
    seed: int = 42
    eval_every: int = 5
    out_dir: str = "results"
