import ffmpeg
import os
import shutil
from datetime import datetime
import glob
from mp4_maker_fetch_music import trim_audio_to_exact_length
import mp4_maker_random_rfm_selector
import time
import textwrap

from openai_utils import summarize_and_estimate_cost 



def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def get_image_files(folder_path):
    image_extensions = ('.jpg', '.png', '.jpeg')
    return sorted(
        (file for file in glob.iglob(os.path.join(folder_path, '*'))
         if file.lower().endswith(image_extensions)),
        key=lambda x: os.path.basename(x).lower()
    )

def create_captioned_images(image_files, captions, image_output_dir, video_width, video_height, caption_props):
    # Calculate max text width for wrapping (a bit less than video width to add padding)
    max_text_width = int(video_width * 0.95)

    for idx, (image_path, caption_text) in enumerate(zip(image_files, captions)):
        # Print message to console
        print(f"Applying caption '{caption_text}' to the image {os.path.basename(image_path)}")

        wrapped_caption_text = textwrap.fill(caption_text, width=50)  # Wrap text after 50 characters. Adjust as needed.

        filename = os.path.basename(image_path)
        new_filename = f'image{idx:04d}{os.path.splitext(filename)[1]}'
        new_filepath_with_caption = os.path.join(image_output_dir, new_filename)

        # Apply filters to scale and add padded background
        video_filter = (
            ffmpeg
            .input(image_path)
            .filter('scale', width=video_width, height=video_height, force_original_aspect_ratio='decrease')
            .filter('pad', width=video_width, height=video_height, x='(ow-iw)/2', y='(oh-ih)/2', color='black')
        )
        
        # Apply the drawtext filter with text wrapping
        video_filter = video_filter.filter(
            'drawtext',
            text=wrapped_caption_text,
            fontcolor=caption_props.get('font_color', 'white'),
            fontsize=caption_props.get('font_size', 36),
            x='(w-tw)/2',  # Centered text horizontally
            y=caption_props.get('caption_offset_y', '0.10*h'),  # Positioned text vertically
            box=1,
            boxcolor=caption_props.get('box_color', 'black@0.5'),
            boxborderw=caption_props.get('box_borderw', 5),
            line_spacing=caption_props.get('line_spacing', 10),  # Optional, adjust line spacing if needed
            fix_bounds=True,  # Ensures text remains within bounding box
            # Removed max_text_width, as it's not a valid FFmpeg drawtext option
        )

        # Now output the image with the caption applied
        video_filter.output(new_filepath_with_caption).run(overwrite_output=True)
        

def generate_video_from_images(image_output_dir, audio_file, output_path, display_duration_per_image):
    try:
        framerate = 1.0 / display_duration_per_image
        ext = next((f for f in os.listdir(image_output_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))), None)
        if ext is None:
            print("No images found to create video.")
            return

        ext = os.path.splitext(ext)[1]
        input_pattern = os.path.join(image_output_dir, 'image%04d' + ext)

        input_stream = ffmpeg.input(input_pattern, framerate=framerate, pattern_type='sequence')
        audio_stream = ffmpeg.input(audio_file)
        output_stream = ffmpeg.output(input_stream, audio_stream, output_path, pix_fmt='yuv420p', vcodec='libx264', acodec='aac', shortest=None)
        
        ffmpeg.run(output_stream)
        
    except ffmpeg.Error as e:
        print("An FFmpeg error occurred while creating the video: ", e.stderr)
        exit(1)
    except Exception as e:
        print("An unexpected error occurred while creating the video: ", e)
        exit(1)

def cleanup(image_output_dir, audio_file):
    try:
        if os.path.exists(image_output_dir):
            shutil.rmtree(image_output_dir)
        if os.path.exists(audio_file):    
            os.remove(audio_file)
    except OSError as e:
        print(f"An error occurred while cleaning up files: {e.strerror}")
        exit(1)

def main(captions_list, working_directory, video_width, video_height, caption_properties, display_duration_per_image, track_type, output_filename_pattern):    
    
    start_time = time.time()  # Start timing the script execution
    summary_data = {
        'audio_file': None,
        'output_file': None,
    }
    # First check if the working directory exists
    if not os.path.exists(working_directory):
        print(f"Error: Working directory '{working_directory}' does not exist.")
        exit(1)

    # Now check if there are any images in the directory
    image_files = get_image_files(working_directory)

    # Debugging: Print captions and associated images to check ordering
    for i, (caption, image_file) in enumerate(zip(captions_list, image_files)):
        print(f"Caption {i}: {caption} -> Image File: {os.path.basename(image_file)}")

    if not image_files:
        print(f"Error: No image files found in '{working_directory}'. Please make sure image files exist.")
        exit(1)

    # Check that the number of captions matches the number of images
    if len(captions_list) != len(image_files):
        print(f"Warning: Number of captions ({len(captions_list)}) does not match the number of images ({len(image_files)}).")
        # Here you'd want to ideally handle what happens if they don't match.
        # For now, we will proceed with the minimum number of available images or captions.
        min_count = min(len(captions_list), len(image_files))
        captions_list = captions_list[:min_count]
        image_files = image_files[:min_count]


    # Create captioned images directory inside working directory
    captioned_images_directory = os.path.join(working_directory, 'captioned_video_images')
    os.makedirs(captioned_images_directory, exist_ok=True)
    #assert len(captions_list) == len(image_files), "Number of captions does not match the number of images."
    video_length_in_seconds = len(image_files) * display_duration_per_image

    #get audio infos
    track_info = mp4_maker_random_rfm_selector.get_rndm_yt_rfm(video_length_in_seconds, track_type=track_type)    
    audio_file = track_info['file_path']
    summary_data['audio_file'] = audio_file


    if not trim_audio_to_exact_length(audio_file, video_length_in_seconds):
        print("Unable to trim audio to exact length. Please check the audio file.")
        exit(1)

    create_captioned_images(image_files, captions_list, captioned_images_directory, video_width, video_height, caption_properties)
    # Create the output file path
    output_file = f'{get_timestamp()}_{output_filename_pattern}.mp4'
    output_path = os.path.join(working_directory, output_file)

    # Generate the video, afterwards we have a complete video length
    generate_video_from_images(captioned_images_directory, audio_file, output_path, display_duration_per_image)
    
    # Calculate total video length using 'display_duration_per_image' and the total number of images
    total_video_length = display_duration_per_image * len(image_files)  # in seconds

    # Before calling cleanup, display all info for the summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    number_of_images = len(image_files)
    video_size = f"{video_width}x{video_height}"
    summary_data['output_file'] = output_file


    summary_data_example = {
        'summary_text': 'Video generation completed.',
        'total_input_tokens': 0,  
        'total_output_tokens': 0,  
        'number_of_images': len(captions_list)
    }
    
    # Get the estimated cost from the function
    estimated_cost = summarize_and_estimate_cost(summary_data_example)
    
    summary = f"""
    ===SUMMARY===
    Execution Time: {elapsed_time:.2f} seconds
    Working Directory: {working_directory}
    Number of Images: {number_of_images}
    Total Video Length: {total_video_length} seconds
    Video Dimensions: {video_size}
    Caption Properties: {caption_properties}
    Display Duration per Image: {display_duration_per_image} seconds
    Audio Track Type: {track_type}
    Audio Track Title: {track_info['title']}
    Audio Track Link: {track_info['link']}
    Audio Track Length: {track_info['length']} seconds
    Audio File: {summary_data['audio_file']}
    Output Filename: {summary_data['output_file']}
    Estimated cost of all OpenAI calls: ${estimated_cost:.2f}
    """

    print(summary.strip())

    cleanup(captioned_images_directory, audio_file)

if __name__ == '__main__':
    print("This script is being run directly. Please use feeder.py to provide input captions.")