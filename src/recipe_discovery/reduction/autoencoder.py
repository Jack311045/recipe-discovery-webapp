"""Deep autoencoder for nonlinear dimensionality reduction."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from recipe_discovery.reduction.base import BaseReducer
from recipe_discovery.reduction.config import AutoencoderConfig
from recipe_discovery.reduction.utils import normalize_embeddings

logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    """Deep fully connected autoencoder with bottleneck."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        
        # 768 -> 512 -> 256 -> 128
        h_dims = config.hidden_dims
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, h_dims[0]),
            nn.BatchNorm1d(h_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),  # Early-layer dropout
            
            nn.Linear(h_dims[0], h_dims[1]),
            nn.BatchNorm1d(h_dims[1]),
            nn.ReLU(),
            
            nn.Linear(h_dims[1], h_dims[2]),
            nn.BatchNorm1d(h_dims[2]),
            nn.ReLU(),
            
            # BOTTLENECK (no activation)
            nn.Linear(h_dims[2], config.latent_dim),
        )
        
        # Decoder (no batch norm to avoid reconstruction artifacts)
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, h_dims[2]),
            nn.ReLU(),
            
            nn.Linear(h_dims[2], h_dims[1]),
            nn.ReLU(),
            
            nn.Linear(h_dims[1], h_dims[0]),
            nn.ReLU(),
            
            nn.Linear(h_dims[0], config.input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input."""
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation."""
        return self.encoder(x)


class AutoencoderReducer(BaseReducer):
    """Reducer wrapper for the Autoencoder module."""

    def __init__(self, config: AutoencoderConfig | None = None) -> None:
        self.config = config or AutoencoderConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = Autoencoder(self.config).to(self.device)
        self.is_fitted = False

    def fit(self, embeddings: np.ndarray) -> None:
        """Training should be handled by train_autoencoder.py. This is a stub."""
        logger.warning("Use scripts/train_autoencoder.py to train the AutoencoderReducer.")
        self.is_fitted = True

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Project embeddings to 2D."""
        if not self.is_fitted:
            logger.warning("AutoencoderReducer may not be fitted. Using random weights.")
        
        normalized = normalize_embeddings(embeddings)
        self.model.eval()
        
        with torch.no_grad():
            tensor_input = torch.tensor(normalized, dtype=torch.float32, device=self.device)
            latent = self.model.encode(tensor_input)
            
        return latent.cpu().numpy()

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        logger.info("Saved AutoencoderReducer weights to %s", path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model weights."""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.is_fitted = True
        self.model.eval()
        logger.info("Loaded AutoencoderReducer weights from %s", path)
