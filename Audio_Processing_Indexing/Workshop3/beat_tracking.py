import librosa.display
import matplotlib.pyplot as plt
import numpy as np

#Load audio
audio_path = "librosa_gallery/examples/audio/Kevin_MacLeod_-_Vibe_Ace.mp3"
y, sr = librosa.load(audio_path)

#Beat tracking
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

#Handle numpy array return
if isinstance(tempo, (list, tuple, np.ndarray)):
    tempo = tempo[0]

print(f"Estimated tempo: {tempo:.2f} BPM")

#Convert frame indices to time (seconds) ===
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

#Plot waveform and beats
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr, alpha=0.6)
plt.vlines(beat_times, -1, 1, color="r", alpha=0.8, linestyle='--', label='Beats')
plt.title(f"Waveform with Beat Tracking\nEstimated Tempo: {tempo:.2f} BPM")
plt.xlabel("Time (s)")
plt.legend()
plt.tight_layout()
plt.savefig("vibeace_beats.png", dpi=300)
plt.show()
