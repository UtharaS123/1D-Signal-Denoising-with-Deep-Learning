"""
autoencoder.py
Denoising Autoencoder (DAE) for 1D signal denoising.
Symmetric encoder-decoder; no skip connections.
"""

import torch.nn as nn


class DenoisingAutoencoder(nn.Module):
    def __init__(self, signal_length=1024):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1,   32,  kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32,  64,  kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64,  128, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, 2, stride=2),
            nn.Conv1d(1,   1,  kernel_size=9, padding=4),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
