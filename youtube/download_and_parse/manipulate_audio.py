from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import random

# Path to the input MP4 file and output MP3 file
input_path = "input.mp4"
output_path = "output_silly.mp3"

# Load the video file and extract the audio
video = VideoFileClip(input_path)
audio = video.audio

# Write the original audio to a temporary file
original_audio_path = "temp_original_audio.mp3"
audio.write_audiofile(original_audio_path)

# Load the original audio with pydub
original_audio = AudioSegment.from_file(original_audio_path)

# Make the audio sound silly by applying randomness
silly_audio = AudioSegment.empty()
min_segment_length = 500  # minimum segment length in ms

while len(silly_audio) < len(original_audio):
    # Randomly select the next segment length without exceeding original length
    max_segment_length = random.randint(min_segment_length, 2000)
    next_seg_length = min(max_segment_length, len(original_audio) - len(silly_audio))
    
    # Get the next segment from the original audio
    segment = original_audio[len(silly_audio):len(silly_audio) + next_seg_length]

    # Randomly reverse the segment
    if random.choice([True, False]):
        segment = segment.reverse()

    # Apply a pitch shift. Instead of changing the speed, we'll just change the pitch
    semitone_step = random.choice([0, 1, -1, 2, -2, 3, -3])
    segment_shifted = segment._spawn(segment.raw_data, overrides={
        "frame_rate": int(segment.frame_rate * 2**(semitone_step / 12.0))
    }).set_frame_rate(segment.frame_rate)
    
    # Randomly vary the volume of the segment
    volume_change = random.randint(-6, 6)  # Change in dB
    segment_shifted += volume_change

    silly_audio += segment_shifted

# Ensure the silly audio is the same length as the original
silly_audio = silly_audio[:len(original_audio)]

# Save the silly audio file
silly_audio.export(output_path, format="mp3")

# Clean up
video.close()
print("Silly audio has been generated at:", output_path)