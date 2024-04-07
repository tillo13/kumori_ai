import os
import glob
from datetime import datetime
from moviepy.editor import ImageSequenceClip, AudioFileClip
import shutil

### GLOBAL VARIABLES ###
FRAMES_DIR_SUFFIX = "_processed"  # Suffix to find the latest processed frames directory
OUTPUT_VIDEO_PREFIX = "final_video"
FPS = 30                          # Frame rate for video output (standard for smooth playback)

# Locate the latest processed directory based on the global suffix
directories = sorted([d for d in os.listdir('.') if d.endswith(FRAMES_DIR_SUFFIX)], key=os.path.getmtime)
if not directories:
    raise Exception("No processed directories found for input frames.")
INPUT_FRAMES_DIR = directories[-1]  # Use the last modified directory

# Locate the most recently downloaded video file for the original audio
video_files = sorted(glob.glob('*.mp4'), key=os.path.getmtime)
if not video_files:
    raise Exception("No downloaded video files found for the audio source.")
AUDIO_FILE = video_files[-1]  # Use the last downloaded .mp4 file that includes audio

# Generate the output video filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_VIDEO = f"{timestamp}_{OUTPUT_VIDEO_PREFIX}.mp4"

def generate_video(image_folder, audio_file, output_video, fps):
    # Begin summary section
    print("===BEGINNING SUMMARY===")
    print(f"Input frames directory: {image_folder}")
    print(f"Audio source: {audio_file}")
    print(f"Output video file: {output_video}")
    print(f"FPS for video output: {fps}")
    print("-" * 30)

    extensions = ['.jpg', '.jpeg', '.png', '.gif']  # Allowed image extensions
    image_files = sorted(
        [img for img in glob.glob(os.path.join(image_folder, '*')) if os.path.splitext(img)[1].lower() in extensions],
        key=os.path.basename
    )

    if not image_files:
        raise Exception(f"No image files found in {image_folder} with allowed extensions {extensions}")

    audio_clip = AudioFileClip(audio_file)
    img_duration = audio_clip.duration / len(image_files)
    video_clip = ImageSequenceClip(image_files, fps=fps)
    video_clip = video_clip.set_audio(audio_clip)

    video_clip.write_videofile(output_video, codec='libx264', fps=fps, audio_codec='aac')

    # End summary section
    print("===SUMMARY===")
    print(f"Total images: {len(image_files)}")
    print(f"Duration per image: {img_duration:.3f} seconds")
    print(f"Final video duration: {audio_clip.duration:.2f} seconds (same as audio duration)")
    print(f"Video with audio saved to {output_video}")
    print("-" * 30)

def move_assets_to_single_folder(video_file, output_video):
    # Extract video ID from the original video filename
    video_id = video_file.split('_')[0]

    # Generate the timestamped directory name
    current_datetime_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination_folder = f"{video_id}_{current_datetime_stamp}"

    # Create and move all related assets to the new directory
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for folder in os.listdir('.'):
        if folder.endswith('_frames') or folder.endswith('_processed'):
            shutil.move(folder, os.path.join(destination_folder, folder))

    # Move the original video file with audio into the consolidated assets folder
    shutil.move(video_file, destination_folder)

    # Move the output video file to the same directory
    shutil.copy(output_video, destination_folder)
    
    print(f"All assets moved to folder: {destination_folder}")

# Generate the video
if __name__ == "__main__":
    print("Starting to create video...")
    generate_video(INPUT_FRAMES_DIR, AUDIO_FILE, OUTPUT_VIDEO, FPS)
    print("Video creation process completed.")
    
    # Move all assets except the final output video
    move_assets_to_single_folder(AUDIO_FILE, OUTPUT_VIDEO)
    
    # The final output video remains in the root, and a copy goes into the assets folder.