import argparse
import numpy as np
import soundfile as sf


def frame_signal(y, win_size, hop_size):
    """
    Cut the signal into frames (like sliding windows).
    If not enough samples at the end, pad zeros.
    Returns: array of shape (#frames, win_size)
    """
    n = len(y)
    if n < win_size:
        # just one short frame
        pad = np.zeros(win_size - n)
        return np.expand_dims(np.concatenate([y, pad]), 0)

    n_frames = int(np.ceil((n - win_size) / hop_size)) + 1
    pad_len = (n_frames - 1) * hop_size + win_size - n
    y_padded = np.pad(y, (0, pad_len))
    idx = np.arange(win_size)[None, :] + np.arange(n_frames)[:, None] * hop_size
    return y_padded[idx]


def compute_band_edges(sr, n_bands, fmax):
    """
    Make evenly spaced band edges from 0 to fmax.
    e.g. for 8 bands and 16kHz: [0,1k,2k,...,8k]
    """
    return np.linspace(0, fmax, n_bands + 1)


def band_energy_from_spectrum(pow_spec, sr, win_size, band_edges_hz):
    """
    Sum energy (power) inside each frequency band.
    pow_spec: one frame power spectrum from rfft
    Returns: 1D array of band energies
    """
    freqs = np.fft.rfftfreq(win_size, 1 / sr)
    n_bands = len(band_edges_hz) - 1
    energies = np.zeros(n_bands)

    for i in range(n_bands):
        f1, f2 = band_edges_hz[i], band_edges_hz[i + 1]
        # last band includes upper edge
        if i == n_bands - 1:
            mask = (freqs >= f1) & (freqs <= f2)
        else:
            mask = (freqs >= f1) & (freqs < f2)
        energies[i] = np.sum(pow_spec[mask])

    return energies


def compute_8band_energies(y, sr,
                           win_size=512,
                           hop_size=512,
                           n_bands=8,
                           fmax=8000.0):
    """
    Main function for assignment 2a.
    Compute 8-band energy per frame.
    """
    if sr != 16000:
        raise ValueError("Expected 16kHz sample rate.")
    if y.ndim != 1:
        raise ValueError("Input must be mono signal.")

    frames = frame_signal(y, win_size, hop_size)
    window = np.hanning(win_size)
    band_edges = compute_band_edges(sr, n_bands, fmax)

    out = np.zeros((frames.shape[0], n_bands))
    for i, frame in enumerate(frames):
        x = frame * window
        X = np.fft.rfft(x)
        pow_spec = np.abs(X) ** 2
        pow_spec /= np.sum(window ** 2) + 1e-10  # normalize
        out[i] = band_energy_from_spectrum(pow_spec, sr, win_size, band_edges)

    return out


def write_energies_txt(energies, out_path):
    """
    Save energy values to a text file.
    Each row = one frame, 8 float values.
    """
    np.savetxt(out_path, energies, fmt="%.6e")


def run_assignment_2a(wav_path, out_txt_path):
    """
    End-to-end process:
      1. Read wav file
      2. Compute 8-band energies
      3. Write results
    """
    y, sr = sf.read(wav_path)
    if y.ndim == 2:
        y = y.mean(axis=1)

    energies = compute_8band_energies(y, sr)
    write_energies_txt(energies, out_txt_path)
    print(f"Done! {energies.shape[0]} frames x {energies.shape[1]} bands written to {out_txt_path}")


def _build_argparser():
    p = argparse.ArgumentParser(description="Assignment 2a: 8-band energies (0–8kHz) @16kHz.")
    p.add_argument("--wav", required=True, help="Path to input WAV (mono, 16kHz).")
    p.add_argument("--out", default="piano_energies.txt", help="Output text file.")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    run_assignment_2a(args.wav, args.out)
