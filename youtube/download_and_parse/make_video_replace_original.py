from moviepy.editor import CompositeVideoClip, VideoFileClip, ImageSequenceClip
import os
import glob

def generate_video(image_folder, input_video, output_video):
    # Load image files, sorted in ascending order and filtering out non-image files
    extensions = ['.jpg', '.jpeg', '.png', '.gif']  # Add other image extensions if needed
    image_files = sorted(
        [img for img in glob.glob(os.path.join(image_folder, '*')) 
         if os.path.splitext(img)[1].lower() in extensions],
        key=os.path.getmtime  # sort files by modification time
    )

    # Load the input video file
    input_video_clip = VideoFileClip(input_video)

    # Calculate time per image
    img_duration = input_video_clip.duration / len(image_files)

    # Create video clip from images
    image_video_clip = ImageSequenceClip(image_files, durations=[img_duration]*len(image_files))

    # Make their duration equal
    image_video_clip = image_video_clip.subclip(0, input_video_clip.duration)

    # Combine the video clips
    video_clip = CompositeVideoClip([input_video_clip, image_video_clip.set_position(('center', 'bottom'))])

    # Write the result to a file
    video_clip.write_videofile(output_video, codec='libx264', fps=24)

    print(f'Video created and saved to {output_video}')

# Parameters
ASSETS_FOLDER = 'assets'
INPUT_VIDEO = os.path.join(ASSETS_FOLDER, 'input.mp4') 
OUTPUT_VIDEO = os.path.join(ASSETS_FOLDER, 'final_output.mp4')

# Create video
generate_video(ASSETS_FOLDER, INPUT_VIDEO, OUTPUT_VIDEO)