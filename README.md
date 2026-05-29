# 1D Signal Denoising with Deep Learning

Deep learning approaches to recovering clean signals from noisy
broadband waveforms, benchmarked against classical methods.
Motivated by the signal denoising challenge in proton-acoustic
imaging for radiotherapy dosimetry.

## Motivation
Proton-acoustic imaging generates weak, broadband acoustic signals
corrupted by noise. This project explores data-driven denoising
architectures that can generalise to such signals, comparing:
- Classical Wiener filter (baseline)
- Denoising Autoencoder (DAE)
- 1D U-Net

## Project Structure
```
signal-denoising/
├── data/
│   └── generate_signals.py     # Synthetic signal generator
├── models/
│   ├── autoencoder.py          # Denoising autoencoder
│   └── unet1d.py               # 1D U-Net
├── train.py                    # Training loop
├── evaluate.py                 # Metrics + plots
├── requirements.txt
└── README.md
```

## Quickstart
```bash
pip install -r requirements.txt
python data/generate_signals.py      # generate dataset
python train.py --model unet         # train 1D U-Net
python train.py --model autoencoder  # train DAE
python evaluate.py                   # compare all methods
```

## References
- Assmann et al. (2015). Ionoacoustic characterization of the
  Bragg peak. Medical Physics, 42(2).
- Ronneberger et al. (2015). U-Net: Convolutional Networks for
  Biomedical Image Segmentation. MICCAI.
