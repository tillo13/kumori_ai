import subprocess
from mutagen.mp3 import MP3
import mp4_maker_random_rfm_selector 
import os

# ===SAMPLE CONFIG VALUES===
EXACT_LENGTH = 10  # length in seconds
MIN_LENGTH = EXACT_LENGTH  # Ensuring the minimum length is at least the exact length

# ===END OF SAMPLE CONFIG VALUES===

def trim_audio_to_exact_length(filename, target_length):
    file_extension = os.path.splitext(filename)[1].lower()
    
    if file_extension == ".mp3":
        # If the file is MP3, we use Mutagen to check its length
        audio = MP3(filename)
        audio_length = int(audio.info.length)
    elif file_extension == ".mp4":
        # If the file is MP4, we use ffprobe to check its length
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                 "format=duration", "-of",
                                 "default=noprint_wrappers=1:nokey=1", filename],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        audio_length = float(result.stdout)
    else:
        print(f"Unsupported file format: {file_extension}")
        return False

    if audio_length == target_length:
        return True  # No trimming needed
    elif audio_length > target_length:
        # Trim the audio file to the exact length
        trimmed_filename = f"{os.path.splitext(filename)[0]}_trimmed{file_extension}"
        subprocess.run([
            "ffmpeg", "-i", filename,
            "-ss", "0", "-to", str(target_length),
            "-c", "copy", trimmed_filename,
            "-y"  # Overwrite output files without asking
        ])
        os.remove(filename)  # Remove the original file
        os.rename(trimmed_filename, filename)  # Rename trimmed file to original file name
        return True
    else:
        # Audio is shorter than the target length; can't trim
        return False



# A new function to fetch a track of a specific genre
def get_techno_track(target_length):
    # Assuming `get_rndm_yt_rfm` accepts a parameter `track_type`
    # Replace `'techno'` with the correct genre identifier if needed
    return mp4_maker_random_rfm_selector.get_rndm_yt_rfm(target_length, track_type='techno')

if __name__ == '__main__':
    MIN_LENGTH = 10  # Minimum length for the track
    # Fetch a techno track with at least the minimum specified length in seconds
    result_file = get_techno_track(MIN_LENGTH)
    if result_file:
        was_trimmed = trim_audio_to_exact_length(result_file, MIN_LENGTH)
        if was_trimmed:
            print(f"The audio file {result_file} was successfully trimmed to {MIN_LENGTH} seconds.")
        else:
            print(f"The audio file {result_file} is shorter than the desired exact length of {MIN_LENGTH} seconds.")