import os

def run_spleeter():
    from spleeter.separator import Separator

    #Setup
    os.makedirs("outputs_spleeter", exist_ok=True)

    #Input path
    audio_path = "librosa_gallery/examples/audio/Cheese_N_Pot-C_-_16_-_The_Raps_Well_Clean_Album_Version.mp3"
    print(f"Input audio: {audio_path}")

    #Initialize and separate
    separator = Separator('spleeter:2stems')
    separator.separate_to_file(audio_path, destination="outputs_spleeter")

    print("\n Separation complete! Files saved under outputs_spleeter/")

#Windows multiprocessing guard
if __name__ == "__main__":
    os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"  # optional explicit ffmpeg path
    run_spleeter()
