import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os

#Setup
os.makedirs("outputs3", exist_ok=True)
os.makedirs("figures3", exist_ok=True)

#Load audio
audio_path = "librosa_gallery/examples/audio/Cheese_N_Pot-C_-_16_-_The_Raps_Well_Clean_Album_Version.mp3"
y, sr = librosa.load(audio_path)
print(f"Loaded: {audio_path} | Duration: {len(y)/sr:.2f}s | Sample rate: {sr}")

#Baseline voice separation using HPSS
y_harmonic, y_percussive = librosa.effects.hpss(y)
# Assume vocals mainly in harmonic component
sf.write("outputs3/vocals_baseline.wav", y_harmonic, sr)
sf.write("outputs3/instrumental_baseline.wav", y_percussive, sr)

#Improved version: frequency filtering + HPSS tuning
# Pre-emphasis + smaller kernel to enhance clarity
y_pre = librosa.effects.preemphasis(y)
y_harmonic_improved, y_percussive_improved = librosa.effects.hpss(y_pre, kernel_size=15, margin=(2.0, 1.0))
sf.write("outputs3/vocals_improved.wav", y_harmonic_improved, sr)
sf.write("outputs3/instrumental_improved.wav", y_percussive_improved, sr)

#Visualization
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
librosa.display.waveshow(y, sr=sr, alpha=0.7)
plt.title("Original audio waveform")

plt.subplot(3, 1, 2)
librosa.display.waveshow(y_harmonic, sr=sr, color='g', alpha=0.7)
plt.title("Baseline separated vocals (harmonic)")

plt.subplot(3, 1, 3)
librosa.display.waveshow(y_harmonic_improved, sr=sr, color='lime', alpha=0.7)
plt.title("Improved separated vocals (preemphasis + tuned HPSS)")

plt.tight_layout()
plt.savefig("figures3/vocal_separation_compare.png", dpi=300)
plt.show()

print("Saved results and figure to outputs3/ and figures3/")

#Apply to another track
audio_path2 = "librosa_gallery/examples/audio/track04_sour_le_vent.mp3"
y2, sr2 = librosa.load(audio_path2)
y2_h, y2_p = librosa.effects.hpss(y2)
y2_pre = librosa.effects.preemphasis(y2)
y2_hi, y2_pi = librosa.effects.hpss(y2_pre, kernel_size=15, margin=(2.0, 1.0))
sf.write("outputs3/track04_vocals_baseline.wav", y2_h, sr2)
sf.write("outputs3/track04_vocals_improved.wav", y2_hi, sr2)
print("complete")
