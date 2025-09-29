#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assignment 2b: UDR encoding with optimizable window/bands for piano.

Usage (script):
    python udr_encoding.py --wav piano.wav --out_dir out_udr

This will create:
    out_udr/udr_raw.txt          # step-by-step codes: U/D/R per frame
    out_udr/udr_compressed.txt   # run-length compressed: Ux/Dx/Rx
    out_udr/maxband_indices.txt  # per-frame max-energy band index
    out_udr/energies.npy         # (num_frames, N) band energies (for检查/复现实验)

Default hyperparams are optimized for piano melody UDR:
    - sr_target=16000, resample_if_needed=True
    - win_size=2048, hop_size=512 (75% overlap)
    - bands: log-spaced, N=24, fmin=55 Hz, fmax=4186 Hz (A1–C8)
"""

import os
import argparse
import numpy as np
import soundfile as sf
import math

# -----------------------------
# Utility: framing + windowing
# -----------------------------
def frame_signal(y: np.ndarray, win_size: int, hop_size: int) -> np.ndarray:
    n = len(y)
    if n == 0:
        return np.zeros((0, win_size), dtype=y.dtype)
    n_frames = int(np.ceil((n - win_size) / hop_size)) + 1 if n >= win_size else 1
    pad_len = max(0, (n_frames - 1) * hop_size + win_size - n)
    y_pad = np.pad(y, (0, pad_len), mode="constant")
    idx = np.arange(win_size)[None, :] + np.arange(n_frames)[:, None] * hop_size
    return y_pad[idx]

def hann_window(win_size: int, dtype=np.float64) -> np.ndarray:
    return np.hanning(win_size).astype(dtype)

# -----------------------------
# Band design
# -----------------------------
def edges_linear(fmin: float, fmax: float, N: int) -> np.ndarray:
    return np.linspace(fmin, fmax, N + 1)

def edges_log(fmin: float, fmax: float, N: int) -> np.ndarray:
    # geometric spacing on frequency axis
    return np.geomspace(max(fmin, 1e-6), fmax, N + 1)

def edges_equal_tempered(fmin: float, fmax: float, steps_per_octave: int) -> np.ndarray:
    # edges at equal-tempered steps; you can later group steps to reach target N
    n_steps = int(np.floor(steps_per_octave * math.log2(fmax / fmin)))
    return fmin * (2.0 ** (np.arange(n_steps + 1) / steps_per_octave))

def build_band_edges(banding: str, fmin: float, fmax: float, N: int,
                     steps_per_octave: int = 12) -> np.ndarray:
    """
    banding in {"linear","log","equal_tempered"}
    - linear/log: returns N+1 edges
    - equal_tempered: first builds ET edges, then groups to closest N bands
    """
    if banding == "linear":
        return edges_linear(fmin, fmax, N)
    if banding == "log":
        return edges_log(fmin, fmax, N)
    if banding == "equal_tempered":
        et_edges = edges_equal_tempered(fmin, fmax, steps_per_octave)
        # Group ET steps into N bands as evenly as possible
        if len(et_edges) < N + 1:
            # fall back to log if too few steps in range
            return edges_log(fmin, fmax, N)
        # pick N+1 approximately evenly spaced indices from et_edges
        idx = np.linspace(0, len(et_edges) - 1, N + 1).round().astype(int)
        return et_edges[idx]
    raise ValueError(f"Unknown banding: {banding}")

# -----------------------------
# Spectrum → band energies
# -----------------------------
def band_energy_from_spectrum(pow_spec: np.ndarray, freqs: np.ndarray, band_edges: np.ndarray) -> np.ndarray:
    energies = []
    for b in range(len(band_edges) - 1):
        f_lo, f_hi = band_edges[b], band_edges[b + 1]
        if b < len(band_edges) - 2:
            sel = (freqs >= f_lo) & (freqs < f_hi)
        else:
            sel = (freqs >= f_lo) & (freqs <= f_hi)
        energies.append(np.sum(pow_spec[sel]))
    return np.asarray(energies, dtype=float)

def compute_band_energies(y: np.ndarray, sr: int,
                          win_size: int, hop_size: int,
                          band_edges: np.ndarray,
                          window_type: str = "hann",
                          eps: float = 1e-12) -> np.ndarray:
    frames = frame_signal(y, win_size, hop_size)
    win = hann_window(win_size, frames.dtype) if window_type == "hann" else np.ones(win_size, dtype=frames.dtype)
    freqs = np.fft.rfftfreq(win_size, d=1.0 / sr)
    out = np.zeros((frames.shape[0], len(band_edges) - 1), dtype=float)
    wnorm = np.sum(win ** 2) + eps
    for i, f in enumerate(frames):
        x = f * win
        X = np.fft.rfft(x, n=win_size)
        pow_spec = (np.abs(X) ** 2) / wnorm
        out[i] = band_energy_from_spectrum(pow_spec, freqs, band_edges)
    return out

# -----------------------------
# UDR encoding
# -----------------------------
def udr_stepwise(maxband_idx: np.ndarray) -> list:
    """
    Compare frame i vs i-1:
      U if higher band, D if lower, R if same.
    Returns list of strings each in {"U","D","R"}, length = len(maxband_idx)-1
    """
    codes = []
    for i in range(1, len(maxband_idx)):
        if maxband_idx[i] > maxband_idx[i - 1]:
            codes.append("U")
        elif maxband_idx[i] < maxband_idx[i - 1]:
            codes.append("D")
        else:
            codes.append("R")
    return codes

def run_length_compress(symbols: list) -> list:
    """
    Compress consecutive identical symbols into SYM+count (e.g., UUU->U3).
    Returns list of strings like ["U3","D1","R5",...]
    """
    if not symbols:
        return []
    out = []
    cur = symbols[0]
    cnt = 1
    for s in symbols[1:]:
        if s == cur:
            cnt += 1
        else:
            out.append(f"{cur}{cnt}")
            cur, cnt = s, 1
    out.append(f"{cur}{cnt}")
    return out

# -----------------------------
# I/O helpers
# -----------------------------
def write_list(path: str, items: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(items) + "\n")

def write_indices(path: str, idx: np.ndarray) -> None:
    np.savetxt(path, idx.astype(int), fmt="%d")

# -----------------------------
# Main pipeline
# -----------------------------
def process_wav_for_udr(
    wav_path: str,
    out_dir: str,
    # I/O
    sr_target: int = 16000,
    resample_if_needed: bool = True,
    # windowing
    win_size: int = 2048,
    hop_size: int = 512,
    window_type: str = "hann",
    # bands
    banding: str = "log",   # {"linear","log","equal_tempered"}
    N: int = 24,
    fmin: float = 55.0,
    fmax: float = 4186.0,
):
    os.makedirs(out_dir, exist_ok=True)

    # read audio
    y, sr = sf.read(wav_path, dtype="float64", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)

    # resample if needed (very lightweight linear interpolation)
    if resample_if_needed and sr != sr_target:
        t_old = np.arange(len(y)) / sr
        t_new = np.arange(int(round(len(y) * sr_target / sr))) / sr_target
        y = np.interp(t_new, t_old, y).astype(np.float64, copy=False)
        sr = sr_target
    elif sr != sr_target:
        raise ValueError(f"Sample rate is {sr} Hz but sr_target={sr_target} and resample_if_needed=False.")

    # build bands
    band_edges = build_band_edges(banding, fmin, min(fmax, sr/2.0), N)

    # energies (num_frames, N)
    energies = compute_band_energies(
        y, sr,
        win_size=win_size,
        hop_size=hop_size,
        band_edges=band_edges,
        window_type=window_type
    )

    # per-frame max-energy band indices
    maxband_idx = np.argmax(energies, axis=1)

    # stepwise U/D/R
    udr = udr_stepwise(maxband_idx)

    # compressed Ux/Dx/Rx
    udr_comp = run_length_compress(udr)

    # save artifacts
    np.save(os.path.join(out_dir, "energies.npy"), energies)
    write_indices(os.path.join(out_dir, "maxband_indices.txt"), maxband_idx)
    write_list(os.path.join(out_dir, "udr_raw.txt"), udr)
    write_list(os.path.join(out_dir, "udr_compressed.txt"), udr_comp)

    # return strings for immediate use (e.g., embed into answers.pdf)
    return " ".join(udr), " ".join(udr_comp), band_edges

def _build_argparser():
    p = argparse.ArgumentParser(description="Assignment 2b: UDR encoding with optimized bands for piano.")
    p.add_argument("--wav", required=True, help="Input WAV path (piano.wav).")
    p.add_argument("--out_dir", default="out_udr", help="Output directory.")
    # Optional overrides
    p.add_argument("--win", type=int, default=2048)
    p.add_argument("--hop", type=int, default=512)
    p.add_argument("--banding", type=str, default="log", choices=["linear","log","equal_tempered"])
    p.add_argument("--N", type=int, default=24)
    p.add_argument("--fmin", type=float, default=55.0)
    p.add_argument("--fmax", type=float, default=4186.0)
    p.add_argument("--sr_target", type=int, default=16000)
    p.add_argument("--no_resample", action="store_true")
    return p

if __name__ == "__main__":
    args = _build_argparser().parse_args()
    process_wav_for_udr(
        wav_path=args.wav,
        out_dir=args.out_dir,
        sr_target=args.sr_target,
        resample_if_needed=(not args.no_resample),
        win_size=args.win,
        hop_size=args.hop,
        banding=args.banding,
        N=args.N,
        fmin=args.fmin,
        fmax=args.fmax,
    )
