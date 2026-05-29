"""
unet1d.py
1D U-Net for signal denoising.
Encoder-decoder with skip connections.
Input/output shape: (B, 1, L)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=9):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch,  out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet1D(nn.Module):
    """
    Encoder: 4 levels of downsampling (MaxPool x2)
    Decoder: 4 levels of upsampling with skip connections
    """

    def __init__(self, in_channels=1, base_features=32):
        super().__init__()
        f = base_features

        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f,     f * 2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.enc4 = ConvBlock(f * 4, f * 8)
        self.pool = nn.MaxPool1d(2)

        self.bottleneck = ConvBlock(f * 8, f * 16)

        self.up4  = nn.ConvTranspose1d(f*16, f*8, 2, stride=2)
        self.dec4 = ConvBlock(f * 16, f * 8)
        self.up3  = nn.ConvTranspose1d(f*8,  f*4, 2, stride=2)
        self.dec3 = ConvBlock(f * 8,  f * 4)
        self.up2  = nn.ConvTranspose1d(f*4,  f*2, 2, stride=2)
        self.dec2 = ConvBlock(f * 4,  f * 2)
        self.up1  = nn.ConvTranspose1d(f*2,  f,   2, stride=2)
        self.dec1 = ConvBlock(f * 2,  f)

        self.out = nn.Conv1d(f, in_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)
