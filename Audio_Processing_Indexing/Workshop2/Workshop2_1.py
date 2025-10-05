import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# load piano.wav
y, sr = librosa.load("piano.wav", sr=None, mono=True)  # keep same setting

# 2. calculate STFT (短时傅立叶变换)
D = librosa.stft(y, n_fft=2048, hop_length=512)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# 3. draw the Spectrogram using matplotlib
plt.figure(figsize=(10, 6))
librosa.display.specshow(S_db, sr=sr, hop_length=512,
                         x_axis="time", y_axis="hz", cmap="magma")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram of piano.wav")
plt.tight_layout()

plt.savefig("spectrogram.png", dpi=200)
plt.close()

print("Spectrogram saved as spectrogram.png")
