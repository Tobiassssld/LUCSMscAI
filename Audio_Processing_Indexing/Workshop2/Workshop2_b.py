import os
import argparse
import numpy as np
import soundfile as sf
import math


# -----------------------------
# Basic helpers: framing + windowing
# -----------------------------
def frame_signal(y, win_size, hop_size):
    """Cut signal into frames (pad zeros at the end if needed)."""
    n = len(y)
    if n < win_size:
        pad = np.zeros(win_size - n)
        return np.expand_dims(np.concatenate([y, pad]), 0)
    n_frames = int(np.ceil((n - win_size) / hop_size)) + 1
    pad_len = (n_frames - 1) * hop_size + win_size - n
    y_padded = np.pad(y, (0, pad_len))
    idx = np.arange(win_size)[None, :] + np.arange(n_frames)[:, None] * hop_size
    return y_padded[idx]


def hann_window(win_size):
    """Return a Hann window."""
    return np.hanning(win_size)


# -----------------------------
# Frequency band definitions
# -----------------------------
def edges_linear(fmin, fmax, N):
    """Make evenly spaced frequency bands."""
    return np.linspace(fmin, fmax, N + 1)


def edges_log(fmin, fmax, N):
    """Make logarithmically spaced bands."""
    return np.geomspace(max(fmin, 1e-6), fmax, N + 1)


def edges_equal_tempered(fmin, fmax, steps_per_octave=12):
    """Equal-tempered scale edges (like musical notes)."""
    n_steps = int(np.floor(steps_per_octave * math.log2(fmax / fmin)))
    return fmin * (2.0 ** (np.arange(n_steps + 1) / steps_per_octave))


def build_band_edges(banding, fmin, fmax, N, steps_per_octave=12):
    """Select which band spacing to use."""
    if banding == "linear":
        return edges_linear(fmin, fmax, N)
    elif banding == "log":
        return edges_log(fmin, fmax, N)
    elif banding == "equal_tempered":
        et_edges = edges_equal_tempered(fmin, fmax, steps_per_octave)
        if len(et_edges) < N + 1:
            # not enough steps, fallback
            return edges_log(fmin, fmax, N)
        idx = np.linspace(0, len(et_edges) - 1, N + 1).round().astype(int)
        return et_edges[idx]
    else:
        raise ValueError("Unknown banding type.")


# -----------------------------
# Spectrum → band energies
# -----------------------------
def band_energy_from_spectrum(pow_spec, freqs, band_edges):
    """Sum power values in each frequency band."""
    energies = []
    for i in range(len(band_edges) - 1):
        f1, f2 = band_edges[i], band_edges[i + 1]
        if i == len(band_edges) - 2:
            mask = (freqs >= f1) & (freqs <= f2)
        else:
            mask = (freqs >= f1) & (freqs < f2)
        energies.append(np.sum(pow_spec[mask]))
    return np.array(energies)


def compute_band_energies(y, sr, win_size, hop_size, band_edges, window_type="hann"):
    """Compute band energies for each frame."""
    frames = frame_signal(y, win_size, hop_size)
    win = hann_window(win_size) if window_type == "hann" else np.ones(win_size)
    freqs = np.fft.rfftfreq(win_size, 1.0 / sr)
    out = np.zeros((frames.shape[0], len(band_edges) - 1))
    norm = np.sum(win ** 2) + 1e-10

    for i, frame in enumerate(frames):
        x = frame * win
        X = np.fft.rfft(x)
        pow_spec = np.abs(X) ** 2 / norm
        out[i] = band_energy_from_spectrum(pow_spec, freqs, band_edges)
    return out


# -----------------------------
# Encode movement as U/D/R
# -----------------------------
def udr_stepwise(maxband_idx):
    """
    Compare current frame's max band with the previous one:
    U if higher, D if lower, R if same.
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


def run_length_compress(symbols):
    """Simple run-length compression (e.g., UUU -> U3)."""
    if not symbols:
        return []
    out = []
    cur = symbols[0]
    count = 1
    for s in symbols[1:]:
        if s == cur:
            count += 1
        else:
            out.append(f"{cur}{count}")
            cur, count = s, 1
    out.append(f"{cur}{count}")
    return out


# -----------------------------
# Simple save functions
# -----------------------------
def write_list(path, items):
    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(items) + "\n")


def write_indices(path, idx):
    np.savetxt(path, idx.astype(int), fmt="%d")


# -----------------------------
# Main pipeline
# -----------------------------
def process_wav_for_udr(
    wav_path,
    out_dir,
    sr_target=16000,
    resample_if_needed=True,
    win_size=2048,
    hop_size=512,
    window_type="hann",
    banding="log",
    N=24,
    fmin=55.0,
    fmax=4186.0,
):
    """Main function for Assignment 2b (UDR encoding)."""
    os.makedirs(out_dir, exist_ok=True)

    # 1. read wav
    y, sr = sf.read(wav_path, dtype="float64", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)

    # 2. resample if needed
    if sr != sr_target:
        if resample_if_needed:
            t_old = np.arange(len(y)) / sr
            t_new = np.arange(int(round(len(y) * sr_target / sr))) / sr_target
            y = np.interp(t_new, t_old, y).astype(np.float64)
            sr = sr_target
        else:
            raise ValueError(f"Wrong sr ({sr}), expected {sr_target}.")

    # 3. build band edges
    band_edges = build_band_edges(banding, fmin, min(fmax, sr / 2), N)

    # 4. compute energies
    energies = compute_band_energies(y, sr, win_size, hop_size, band_edges, window_type)
    maxband_idx = np.argmax(energies, axis=1)

    # 5. UDR coding
    udr_raw = udr_stepwise(maxband_idx)
    udr_compressed = run_length_compress(udr_raw)

    # 6. save results
    np.save(os.path.join(out_dir, "energies.npy"), energies)
    write_indices(os.path.join(out_dir, "maxband_indices.txt"), maxband_idx)
    write_list(os.path.join(out_dir, "udr_raw.txt"), udr_raw)
    write_list(os.path.join(out_dir, "udr_compressed.txt"), udr_compressed)

    print("Finished! Results saved in", out_dir)
    return " ".join(udr_raw), " ".join(udr_compressed)


def _build_argparser():
    p = argparse.ArgumentParser(description="Assignment 2b: UDR encoding with customizable bands.")
    p.add_argument("--wav", required=True, help="Input WAV path (mono 16kHz).")
    p.add_argument("--out_dir", default="out_udr", help="Output folder.")
    p.add_argument("--win", type=int, default=2048)
    p.add_argument("--hop", type=int, default=512)
    p.add_argument("--banding", type=str, default="log", choices=["linear", "log", "equal_tempered"])
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
        resample_if_needed=not args.no_resample,
        win_size=args.win,
        hop_size=args.hop,
        banding=args.banding,
        N=args.N,
        fmin=args.fmin,
        fmax=args.fmax,
    )
