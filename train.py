"""
train.py
Training loop for signal denoising models.

Usage:
    python train.py --model unet
    python train.py --model autoencoder --epochs 50 --lr 1e-3
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.unet1d import UNet1D
from models.autoencoder import DenoisingAutoencoder


def load_split(split):
    noisy = torch.from_numpy(np.load(f"data/arrays/{split}_noisy.npy"))
    clean = torch.from_numpy(np.load(f"data/arrays/{split}_clean.npy"))
    return TensorDataset(noisy, clean)


def snr(pred, target):
    signal_power = torch.mean(target ** 2)
    noise_power  = torch.mean((pred - target) ** 2)
    return 10 * torch.log10(signal_power / (noise_power + 1e-10))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = DataLoader(load_split("train"),
                              batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(load_split("val"),
                              batch_size=args.batch_size)

    if args.model == "unet":
        model = UNet1D().to(device)
    else:
        model = DenoisingAutoencoder().to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimiser, patience=5)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimiser.zero_grad()
            pred = model(noisy)
            loss = criterion(pred, clean)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()

        model.eval()
        val_loss, val_snr = 0.0, 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                pred      = model(noisy)
                val_loss += criterion(pred, clean).item()
                val_snr  += snr(pred, clean).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        val_snr    /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d} | "
              f"Train: {train_loss:.6f} | "
              f"Val: {val_loss:.6f} | "
              f"Val SNR: {val_snr:.2f} dB")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       f"checkpoints/{args.model}_best.pt")

    print(f"Done. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    import os
    os.makedirs("checkpoints", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unet",
                        choices=["unet", "autoencoder"])
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
