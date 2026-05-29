"""
generate_signals.py
Synthetic broadband signal generator mimicking proton-acoustic
pulses. Saves train/val/test splits as numpy arrays.
"""

import numpy as np
import os


def gaussian_pulse(t, t0, sigma, amplitude=1.0):
    """Gaussian pulse centred at t0."""
    return amplitude * np.exp(
        -((t - t0) ** 2) / (2 * sigma ** 2))


def generate_sample(length=1024, fs=1e6, snr_db=10.0):
    """
    Generate one (noisy_signal, clean_signal) pair.
    """
    t = np.linspace(0, length / fs, length)

    n_pulses = np.random.randint(1, 4)
    clean = np.zeros(length)
    for _ in range(n_pulses):
        t0        = np.random.uniform(0.2, 0.8) * t[-1]
        sigma     = np.random.uniform(0.02, 0.08) * t[-1]
        amplitude = np.random.uniform(0.5, 1.0)
        clean    += gaussian_pulse(t, t0, sigma, amplitude)

    signal_power = np.mean(clean ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise        = np.random.normal(
                       0, np.sqrt(noise_power), length)
    noisy = clean + noise

    return noisy.astype(np.float32), clean.astype(np.float32)


def build_dataset(n_samples=10000, snr_db=10.0,
                  length=1024, seed=42):
    np.random.seed(seed)
    noisy_all = np.zeros((n_samples, 1, length), dtype=np.float32)
    clean_all = np.zeros((n_samples, 1, length), dtype=np.float32)
    for i in range(n_samples):
        noisy, clean       = generate_sample(length=length,
                                             snr_db=snr_db)
        noisy_all[i, 0, :] = noisy
        clean_all[i, 0, :] = clean
    return noisy_all, clean_all


if __name__ == "__main__":
    os.makedirs("data/arrays", exist_ok=True)
    for split, n, seed in [("train",8000,0),
                            ("val",  1000,1),
                            ("test", 1000,2)]:
        noisy, clean = build_dataset(n_samples=n, seed=seed)
        np.save(f"data/arrays/{split}_noisy.npy", noisy)
        np.save(f"data/arrays/{split}_clean.npy", clean)
        print(f"Saved {split}: {n} samples")
