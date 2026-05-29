"""
evaluate.py
Evaluate all methods (Wiener, DAE, U-Net) on the test set
and produce comparison plots.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import wiener

from models.unet1d import UNet1D
from models.autoencoder import DenoisingAutoencoder


def compute_snr(pred, target):
    sp  = np.mean(target ** 2)
    np_ = np.mean((pred - target) ** 2)
    return 10 * np.log10(sp / (np_ + 1e-10))


def compute_mse(pred, target):
    return np.mean((pred - target) ** 2)


def load_model(model_class, path, device):
    m = model_class().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


def run_dl_model(model, noisy_np, device):
    x = torch.from_numpy(noisy_np[:, np.newaxis, :]).to(device)
    with torch.no_grad():
        out = model(x).cpu().numpy()[:, 0, :]
    return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    noisy = np.load("data/arrays/test_noisy.npy")[:, 0, :]
    clean = np.load("data/arrays/test_clean.npy")[:, 0, :]

    wiener_out = np.array([wiener(s) for s in noisy])

    unet     = load_model(UNet1D, "checkpoints/unet_best.pt", device)
    dae      = load_model(DenoisingAutoencoder,
                          "checkpoints/autoencoder_best.pt", device)
    unet_out = run_dl_model(unet, noisy, device)
    dae_out  = run_dl_model(dae,  noisy, device)

    results = {
        "Noisy input": (noisy,      "grey"),
        "Wiener":      (wiener_out, "orange"),
        "DAE":         (dae_out,    "green"),
        "1D U-Net":    (unet_out,   "royalblue"),
    }

    print(f"\n{'Method':<18} {'SNR (dB)':>10} {'MSE':>12}")
    print("-" * 42)
    for name, (pred, _) in results.items():
        s = np.mean([compute_snr(pred[i], clean[i])
                     for i in range(len(clean))])
        m = np.mean([compute_mse(pred[i], clean[i])
                     for i in range(len(clean))])
        print(f"{name:<18} {s:>10.2f} {m:>12.6f}")
