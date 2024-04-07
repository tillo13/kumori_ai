### GLOBAL VARIABLES ###
YOUTUBE_URL = "https://www.youtube.com/watch?v=fvhOpjK5ofU"
YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

FRAMES_PER_SECOND = 24
OUTPUT_DIRECTORY_NAME = "parsed_frames"
PERCENTAGE_OF_ORIGINAL = 100

import ffmpeg
import os
from pytube import YouTube
from datetime import datetime
import time
from PIL import Image
import io
import os
import ffmpeg
from PIL import Image
from datetime import datetime
import tempfile

def parse_video(input_file, output_dir, percentage):
    # Ensure the output directory name is lowercase
    output_dir = output_dir.casefold()

    # Check if the output directory exists, create if absent
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate a timestamp for the frames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Run ffmpeg pipeline to extract frames to temporary directory
        (
            ffmpeg
            .input(input_file)
            .filter('fps', fps=FRAMES_PER_SECOND)
            .output(os.path.join(temp_dir, 'frame%04d.png'), vsync='vfr')
            .run()
        )

        # List of temporary frames
        frames = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.png')]
        total_frames = len(frames)  # Determine the total number of frames
        
        frame_count = 0

        for i, frame_path in enumerate(frames):
            try:
                img = Image.open(frame_path)

                # Calculate new size based on the percentage
                new_width = int(img.width * percentage / 100)
                new_height = int(img.height * percentage / 100)
                
                # Resize the image
                img_resized = img.resize((new_width, new_height), Image.BILINEAR)

                # Define new frame filename with timestamp
                frame_filename = os.path.join(output_dir, f'{timestamp}_frame{i:04d}.png')

                # Save the resized frame
                img_resized.save(frame_filename)
                frame_count += 1

                # Print the progress statement
                print(f"Saving frame {i + 1} of {total_frames}...")

            except Exception as e:
                print(f"Error resizing frame {i+1}: {e}")

        return frame_count

def download_video(url):
    """
    Downloads the highest quality video from the given URL,
    names it after the video ID, and returns the filename.
    """
    yt = YouTube(url)

    # Get the YouTube video ID from the URL
    video_id = yt.video_id

    # Select the highest quality stream available
    video_stream = yt.streams.filter(progressive=True, file_extension='mp4').get_highest_resolution()
    
    if video_stream:
        # Set the filename for the downloaded video using video ID
        file_name = f"{video_id}.mp4"
        
        # Download the video
        video_stream.download(filename=file_name)
        
        return file_name
    else:
        raise Exception(f"No mp4 video streams found for {url}")

if __name__ == "__main__":
    start_time = time.time()
    try:
        yt = YouTube(YOUTUBE_URL)

        # Retrieve and print video details
        video_details = {
            'Title': yt.title,
            'Length (s)': yt.length,
            'Publish Date': yt.publish_date.strftime('%Y-%m-%d') if yt.publish_date else 'Unknown',
            'Views': f"{yt.views:,}",
            'Author': yt.author,
            'Resize Percentage': f"{PERCENTAGE_OF_ORIGINAL}%",
        }

        print("==VIDEO DETAILS==")
        print(f"Resizing frames to {PERCENTAGE_OF_ORIGINAL}% of the original size")
        for detail, value in video_details.items():
            print(f"{detail}: {value}")

        # Proceed with the download and parsing
        downloaded_file_name = download_video(YOUTUBE_URL)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_directory_name = f"{timestamp}_{OUTPUT_DIRECTORY_NAME}"

        number_of_frames = parse_video(downloaded_file_name, output_directory_name, PERCENTAGE_OF_ORIGINAL)
        
        duration = time.time() - start_time
        
        # Print out summary of the process
        print("==SUMMARY==")
        print(f"Video downloaded: {downloaded_file_name}")
        print(f"Total frames extracted and resized to {PERCENTAGE_OF_ORIGINAL}%: {number_of_frames}")
        print(f"Elapsed time: {duration:.2f} seconds")
        print("Video processing complete. Frames extracted and resized.")
    except Exception as e:
        print(str(e))