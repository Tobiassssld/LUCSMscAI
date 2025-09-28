#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assignment 2a: Compute 8-band energies per 512-sample window for a 16kHz mono WAV.

Usage (script):
    python compute_piano_energies.py --wav piano.wav --out piano_energies.txt

If you run in Jupyter, just call:
    run_assignment_2a("piano.wav", "piano_energies.txt")
"""

import argparse
import numpy as np
import soundfile as sf  # pip install soundfile

def frame_signal(y: np.ndarray, win_size: int, hop_size: int) -> np.ndarray:
    """
    Slice 1D signal y into frames of length win_size with hop hop_size.
    Pad end with zeros to include last partial frame.
    Returns shape: (num_frames, win_size)
    """
    n = len(y)
    if n == 0:
        return np.zeros((0, win_size), dtype=y.dtype)

    n_frames = int(np.ceil((n - win_size) / hop_size)) + 1 if n >= win_size else 1
    pad_len = max(0, (n_frames - 1) * hop_size + win_size - n)
    y_pad = np.pad(y, (0, pad_len), mode="constant")
    idx = np.arange(win_size)[None, :] + np.arange(n_frames)[:, None] * hop_size
    return y_pad[idx]

def compute_band_edges(sr: int, n_bands: int, fmax: float) -> np.ndarray:
    """
    Evenly-spaced band edges from 0 to fmax (inclusive).
    Returns array of length n_bands+1, e.g. [0, 1000, 2000, ..., 8000]
    """
    return np.linspace(0.0, fmax, n_bands + 1)

def band_energy_from_spectrum(pow_spec: np.ndarray, sr: int, win_size: int,
                              band_edges_hz: np.ndarray) -> np.ndarray:
    """
    Sum power within each band (half-spectrum).
    pow_spec: shape (N_freq,), N_freq = win_size//2 + 1 from rfft
    band_edges_hz: array of band boundaries in Hz, len = n_bands + 1
    Returns energies: shape (n_bands,)
    """
    n_bins = pow_spec.shape[0]
    freqs = np.fft.rfftfreq(win_size, d=1.0/sr)  # shape (n_bins,)

    energies = []
    for b in range(len(band_edges_hz) - 1):
        f_lo, f_hi = band_edges_hz[b], band_edges_hz[b+1]
        # include left edge, exclude right edge except last band includes right edge
        if b < len(band_edges_hz) - 2:
            sel = (freqs >= f_lo) & (freqs < f_hi)
        else:
            sel = (freqs >= f_lo) & (freqs <= f_hi)
        energies.append(np.sum(pow_spec[sel]))
    return np.asarray(energies, dtype=float)

def compute_8band_energies(y: np.ndarray, sr: int,
                           win_size: int = 512,
                           hop_size: int = 512,
                           n_bands: int = 8,
                           fmax: float = 8000.0,
                           window_type: str = "hann",
                           eps: float = 1e-12) -> np.ndarray:
    """
    Core routine for Assignment 2a.

    - y       : mono waveform (float32 or float64), any amplitude range
    - sr      : sample rate, must be 16000 for exact band layout
    - win/hop : 512-sample frames, non-overlapping per spec
    - n_bands : 8 bands
    - fmax    : 8000 Hz (Nyquist for 16kHz)
    - window  : apply Hann window to reduce spectral leakage

    Returns: array shape (num_frames, n_bands)
    """
    if sr != 16000:
        # 作业假设 16kHz。如遇不同采样率，你可在 2b 优化里选择重采样或相应调整 fmax/bands。
        raise ValueError(f"Sample rate must be 16000 Hz for Assignment 2a. Got {sr}.")

    if y.ndim != 1:
        raise ValueError("Input signal must be mono (1D).")

    frames = frame_signal(y, win_size, hop_size)  # (T, win_size)

    if window_type == "hann":
        win = np.hanning(win_size).astype(frames.dtype)
    elif window_type is None:
        win = np.ones(win_size, dtype=frames.dtype)
    else:
        raise ValueError(f"Unsupported window_type: {window_type}")

    band_edges = compute_band_edges(sr, n_bands, fmax)  # [0,1k,...,8k] for 16k

    out = np.zeros((frames.shape[0], n_bands), dtype=float)

    for i, f in enumerate(frames):
        x = f * win
        # real FFT, keep non-negative freqs
        X = np.fft.rfft(x, n=win_size)  # (win_size//2 + 1,)
        pow_spec = (np.abs(X) ** 2) / (np.sum(win**2) + eps)  # window-energy normalization
        out[i] = band_energy_from_spectrum(pow_spec, sr, win_size, band_edges)

    return out

def write_energies_txt(energies: np.ndarray, out_path: str, fmt: str = "%.6e") -> None:
    """
    Write per-frame band energies to text file.
    Each line: 8 space-separated floating-point numbers (one frame).
    """
    if energies.ndim != 2:
        raise ValueError("energies must be 2D: (num_frames, n_bands)")
    np.savetxt(out_path, energies, fmt=fmt)

def run_assignment_2a(wav_path: str, out_txt_path: str) -> None:
    """
    End-to-end for Assignment 2a:
    - load wav
    - (assert) 16kHz, mono
    - compute 8-band energies per 512-sample window
    - write piano_energies.txt
    """
    # soundfile preserves original dtype; convert to float64 for numeric stability
    y, sr = sf.read(wav_path, dtype="float64", always_2d=False)

    # If file is stereo, average to mono (defensive, though spec says mono)
    if y.ndim == 2:
        y = y.mean(axis=1)

    # Scale 8-bit PCM to [-1,1] if needed (sf.read already returns float in [-1,1] usually)
    # Here we assume float in [-1,1]; if you detect int8, convert as y = y.astype(np.float64) / 128

    energies = compute_8band_energies(
        y, sr,
        win_size=512,
        hop_size=512,
        n_bands=8,
        fmax=8000.0,
        window_type="hann"
    )
    write_energies_txt(energies, out_txt_path)
    print(f"[OK] Wrote {energies.shape[0]} frames × {energies.shape[1]} bands to: {out_txt_path}")

def _build_argparser():
    p = argparse.ArgumentParser(description="Assignment 2a: 8-band energies (0–8kHz) per 512-sample window @16kHz.")
    p.add_argument("--wav", required=True, help="Input WAV path (mono, 8-bit, 16kHz expected).")
    p.add_argument("--out", default="piano_energies.txt", help="Output text file path.")
    return p

if __name__ == "__main__":
    args = _build_argparser().parse_args()
    run_assignment_2a(args.wav, args.out)
