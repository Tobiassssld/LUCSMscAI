import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# 1. 加载 piano.wav
y, sr = librosa.load("piano.wav", sr=None, mono=True)  # 保持原始采样率

# 2. 计算 STFT (短时傅立叶变换)
D = librosa.stft(y, n_fft=2048, hop_length=512)  # 频率分辨率高一些
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# 3. 画频谱图
plt.figure(figsize=(10, 6))
librosa.display.specshow(S_db, sr=sr, hop_length=512,
                         x_axis="time", y_axis="hz", cmap="magma")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram of piano.wav")
plt.tight_layout()

# 4. 保存为图像文件
plt.savefig("spectrogram.png", dpi=200)
plt.close()

print("Spectrogram saved as spectrogram.png")
