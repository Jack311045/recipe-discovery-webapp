"""Train the autoencoder projection model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from recipe_discovery.reduction.autoencoder import AutoencoderReducer
from recipe_discovery.reduction.config import AutoencoderConfig
from recipe_discovery.reduction.utils import normalize_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_autoencoder(args: argparse.Namespace) -> None:
    """Run the training loop."""
    logger.info("Loading embeddings from %s", args.embeddings_path)
    if not Path(args.embeddings_path).exists():
        logger.error("Embeddings file not found: %s", args.embeddings_path)
        return

    embeddings = np.load(args.embeddings_path)
    
    # Pre-normalize the entire dataset
    normalized = normalize_embeddings(embeddings)
    
    # Split 80/20 train/val
    n_samples = len(normalized)
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * 0.8)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    train_tensor = torch.tensor(normalized[train_idx], dtype=torch.float32)
    val_tensor = torch.tensor(normalized[val_idx], dtype=torch.float32)
    
    train_loader = DataLoader(
        TensorDataset(train_tensor), 
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor), 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    config = AutoencoderConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        noise_std=args.noise_std,
    )
    
    reducer = AutoencoderReducer(config)
    model = reducer.model
    device = reducer.device
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=config.patience, factor=config.lr_factor
    )
    criterion = nn.MSELoss()
    
    best_val_loss = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "autoencoder_weights.pt"
    
    logger.info("Starting training on %s", device)
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            if args.denoise:
                noise = torch.randn_like(batch) * config.noise_std
                noisy_input = batch + noise
                reconstructed = model(noisy_input)
            else:
                reconstructed = model(batch)
                
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
            
        train_loss /= len(train_idx)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                val_loss += loss.item() * len(batch)
                
        val_loss /= len(val_idx)
        scheduler.step(val_loss)
        
        logger.info(
            "Epoch %d/%d - Train Loss: %.6f - Val Loss: %.6f",
            epoch + 1, config.epochs, train_loss, val_loss
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            reducer.save_checkpoint(weights_path)
            
    logger.info("Training complete. Best val loss: %.6f", best_val_loss)
    
    # Reload best model and generate full projections
    reducer.load_checkpoint(weights_path)
    logger.info("Generating 2D projections for all data...")
    projections_2d = reducer.transform(embeddings)
    
    proj_path = output_dir / "projections_2d.npy"
    np.save(proj_path, projections_2d)
    logger.info("Saved projections to %s", proj_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Autoencoder Reducer")
    parser.add_argument("--embeddings_path", type=str, default="data/artifacts/embeddings.npy")
    parser.add_argument("--output_dir", type=str, default="data/artifacts")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--denoise", action="store_true", help="Enable denoising mode")
    parser.add_argument("--noise_std", type=float, default=0.05)
    
    args = parser.parse_args()
    train_autoencoder(args)


if __name__ == "__main__":
    main()
