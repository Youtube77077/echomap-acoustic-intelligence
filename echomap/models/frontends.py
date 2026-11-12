"""Numpy re-implementations of the model input frontends.

These must match the reference PyTorch frontends exactly. The CI workflow
`convert-models.yml` checks both against the originals (tools/convert/*), and
the parity results are stored next to the exported models (parity.json).

- EfficientAT (Schmid et al.): models/preprocess.py::AugmentMelSTFT in eval mode,
  32 kHz, n_fft 1024, win 800, hop 320, 128 kaldi mel bands from 0 to 15 kHz.
  The mel filterbank matrix is exported from torchaudio by CI (`*_melbank.npy`),
  so it does not need re-deriving here.
- Whisper (OpenAI): 16 kHz, n_fft 400, hop 160, 80 slaney mel bands, log10,
  dynamic-range clamp of 8, then (x + 4) / 4. The input is padded to 30 s.
"""
from __future__ import annotations

import numpy as np


def _stft_power(x: np.ndarray, n_fft: int, hop: int, window: np.ndarray) -> np.ndarray:
    """Power spectrogram with torch.stft(center=True, pad_mode='reflect') semantics.

    `window` must already be zero-padded to n_fft and centred, as torch does.
    Returns an array of shape (n_fft // 2 + 1, n_frames), float32.
    """
    pad = n_fft // 2
    if x.shape[-1] <= pad:
        # reflect padding needs len > pad; extend with zeros (only for tiny inputs)
        x = np.pad(x, (0, pad + 1 - x.shape[-1]))
    xp = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (xp.shape[-1] - n_fft) // hop
    frames = np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::hop][:n_frames]
    spec = np.fft.rfft(frames * window[None, :], n=n_fft, axis=-1)
    power = (spec.real.astype(np.float64) ** 2 + spec.imag.astype(np.float64) ** 2)
    return power.T.astype(np.float32)


def _centered_window(win: np.ndarray, n_fft: int) -> np.ndarray:
    left = (n_fft - win.shape[0]) // 2
    return np.pad(win, (left, n_fft - win.shape[0] - left))


# ----------------------------------------------------------------------------- EfficientAT
EAT_SR = 32000
EAT_N_FFT = 1024
EAT_WIN = 800
EAT_HOP = 320


def hann_symmetric(n: int) -> np.ndarray:
    """torch.hann_window(n, periodic=False)."""
    if n == 1:
        return np.ones(1, dtype=np.float32)
    k = np.arange(n, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * k / (n - 1))).astype(np.float32)


def hann_periodic(n: int) -> np.ndarray:
    """torch.hann_window(n, periodic=True)."""
    k = np.arange(n, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)).astype(np.float32)


def efficientat_mel(wave_32k: np.ndarray, melbank: np.ndarray) -> np.ndarray:
    """Waveform (float32, 32 kHz, mono) -> normalised log-mel of shape (128, T)."""
    x = np.asarray(wave_32k, dtype=np.float32)
    # pre-emphasis: conv1d with kernel [-0.97, 1], no padding  => y[n] = x[n+1] - 0.97 x[n]
    x = x[1:] - np.float32(0.97) * x[:-1]
    win = _centered_window(hann_symmetric(EAT_WIN), EAT_N_FFT)
    power = _stft_power(x, EAT_N_FFT, EAT_HOP, win)  # (513, T)
    mel = melbank.astype(np.float32) @ power          # (128, T)
    mel = np.log(mel + np.float32(1e-5))
    return ((mel + np.float32(4.5)) / np.float32(5.0)).astype(np.float32)


# ----------------------------------------------------------------------------- Whisper
WH_SR = 16000
WH_N_FFT = 400
WH_HOP = 160
WH_CHUNK = 30 * WH_SR  # 480000 samples


def whisper_log_mel(wave_16k: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """Waveform (float32, 16 kHz) -> Whisper log-mel (80, 3000). Pads or trims to 30 s.

    mel_filters: (80, 201) slaney mel matrix exported by CI from the reference
    feature extractor.
    """
    x = np.asarray(wave_16k, dtype=np.float32)
    if x.shape[0] >= WH_CHUNK:
        x = x[:WH_CHUNK]
    else:
        x = np.pad(x, (0, WH_CHUNK - x.shape[0]))
    power = _stft_power(x, WH_N_FFT, WH_HOP, hann_periodic(WH_N_FFT))[:, :-1]  # drop last frame
    mel = mel_filters.astype(np.float32) @ power
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)
