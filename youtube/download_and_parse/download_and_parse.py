import ffmpeg
import os
import datetime
from pytube import YouTube

# Function to handle progress updates
def on_progress(stream, chunk, bytes_remaining):
    total_size = stream.filesize
    bytes_downloaded = total_size - bytes_remaining
    percent_complete = bytes_downloaded / total_size * 100
    print(f"Downloading... {percent_complete:.2f}% complete", end='\r')


# Function to handle completion of the download
def on_complete(stream, file_path):
    print("\nDownload Complete")
    parse_video(file_path)


# Function to parse the video and extract frames
def parse_video(input_file):
    # Output directory
    output_dir = 'parsed_frames'

    # Check if the output directory exists and create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the start time of the video
    start_time = datetime.datetime.now()

    # Format the start time string with underscores instead of colons, dashes and periods
    formatted_start_time = start_time.strftime('%Y_%m_%d_%H_%M_%S_%f')

    # Create and run ffmpeg pipeline
    #fp5 = how many frames per second (5= 5 every second) =--more frames clearly the more smooth the video output may be
    ffmpeg.input(input_file)\
            .filter('fps', fps=5)\
            .output(os.path.join(output_dir, f'{formatted_start_time}%04d.png'))\
            .run()


# Main download logic
if __name__ == "__main__":
    yt_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    yt = YouTube(
            yt_url,
            on_progress_callback=on_progress,
            on_complete_callback=on_complete,
            use_oauth=False,
            allow_oauth_cache=True
        )
    
    # Choose the first mp4 stream available
    video_stream = yt.streams.filter(file_extension='mp4').first()
    
    if video_stream:
        # Assign a file name for the downloaded video
        file_name = f"{yt.title}.mp4"
        # Download the video
        video_stream.download(filename=file_name)
    else:
        print(f"No mp4 video streams found for {yt_url}")