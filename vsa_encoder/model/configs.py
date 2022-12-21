from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    n_features: int = 5
    image_size: Tuple[int, int, int] = (1, 64, 64)
    latent_dim: int = 1024
    bind_mode: str = 'fourier'
    lr: float = 0.00025
    kld_coef: float = 0.001
    monitor: str = 'Validation/Total'


@dataclass
class CheckpointsConfig:
    save_top_k: int = 1
    every_k_epochs: int = 10
    check_val_every_n_epochs: int = 5
    ckpt_path: Optional[str] = None


@dataclass
class DatasetConfig:
    path_to_dataset: str
    mode: str = 'dsprites'


@dataclass
class ExperimentConfig:
    gradient_clip: float = 0.0
    seed: int = 0
    batch_size: int = 64
    steps_per_epoch: int = 0

    devices: int = 0
    max_epochs: int = 400
    accelerator: str = 'gpu'
    profiler: Optional[str] = None


@dataclass
class VAEConfig:
    dataset: DatasetConfig
    experiment: ExperimentConfig
    model: ModelConfig
    checkpoints: CheckpointsConfig
