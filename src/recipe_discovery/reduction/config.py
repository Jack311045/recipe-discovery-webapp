"""Hyperparameters for dimensionality reduction."""

from dataclasses import dataclass


@dataclass
class AutoencoderConfig:
    """Hyperparameters for the deep autoencoder."""

    input_dim: int = 768
    latent_dim: int = 2
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout_rate: float = 0.1
    batch_size: int = 256
    epochs: int = 50
    noise_std: float = 0.05
    patience: int = 5
    lr_factor: float = 0.5
