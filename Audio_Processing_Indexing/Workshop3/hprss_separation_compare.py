# hprss_separation_compare.py
# Assignment 2 – Harmonic–Percussive Source Separation (Comparison Version)
# Author: [Your Name]

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os

# === 0. Setup ===
os.makedirs("outputs", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# === 1. Load audio ===
audio_path = "librosa_gallery/examples/audio/Kevin_MacLeod_-_Vibe_Ace.mp3"
y, sr = librosa.load(audio_path)
print(f"Loaded audio: {audio_path} | Duration: {len(y)/sr:.2f}s | Sample rate: {sr}")

# === 2. Baseline HPSS (default parameters) ===
y_h_base, y_p_base = librosa.effects.hpss(y)

# === 3. Improved HPSS (adjusted kernel + preemphasis) ===
# Pre-emphasis emphasizes high frequencies → clearer percussion
y_pre = librosa.effects.preemphasis(y)
y_h_improved, y_p_improved = librosa.effects.hpss(y_pre, kernel_size=15)

# === 4. Visualization: waveform comparison ===
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
librosa.display.waveshow(y, sr=sr, alpha=0.7)
plt.title("Original waveform")

plt.subplot(4, 1, 2)
librosa.display.waveshow(y_h_base, sr=sr, color='g', alpha=0.7)
plt.title("Baseline Harmonic (default HPSS)")

plt.subplot(4, 1, 3)
librosa.display.waveshow(y_h_improved, sr=sr, color='lime', alpha=0.7)
plt.title("Improved Harmonic (kernel_size=15, preemphasis)")

plt.subplot(4, 1, 4)
librosa.display.waveshow(y_p_improved, sr=sr, color='r', alpha=0.7)
plt.title("Improved Percussive (enhanced transient clarity)")

plt.tight_layout()
plt.savefig("figures/hprss_vibeace_compare.png", dpi=300)
plt.show()
print("✅ Saved figure: figures/hprss_vibeace_compare.png")

# === 5. Save separated audio ===
sf.write("outputs/vibeace_harmonic_baseline.wav", y_h_base, sr)
sf.write("outputs/vibeace_percussive_baseline.wav", y_p_base, sr)
sf.write("outputs/vibeace_harmonic_improved.wav", y_h_improved, sr)
sf.write("outputs/vibeace_percussive_improved.wav", y_p_improved, sr)
print("✅ Saved separated audio files to outputs/")

# === 6. (Optional) Apply to second track ===
audio_path2 = "librosa_gallery/examples/audio/track03_rolling_stone_blues_end.mp3"
y2, sr2 = librosa.load(audio_path2)
y2_h_base, y2_p_base = librosa.effects.hpss(y2)
y2_pre = librosa.effects.preemphasis(y2)
y2_h_improved, y2_p_improved = librosa.effects.hpss(y2_pre, kernel_size=15)

sf.write("outputs/track03_harmonic_baseline.wav", y2_h_base, sr2)
sf.write("outputs/track03_percussive_baseline.wav", y2_p_base, sr2)
sf.write("outputs/track03_harmonic_improved.wav", y2_h_improved, sr2)
sf.write("outputs/track03_percussive_improved.wav", y2_p_improved, sr2)
print("✅ track03 separation complete.")
