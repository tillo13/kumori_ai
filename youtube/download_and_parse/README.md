
# Video Frame Processing Pipeline

This documentation provides an overview of the workflow implemented by `download_and_parse.py` and `recreate_with_ai.py`, which allow for downloading a YouTube video, parsing it into frames, and then using an AI model to transform these frames, creating a stylized version of the original video embodied by a sequence of images.

## download_and_parse.py

The `download_and_parse.py` script is responsible for two primary functions:

1. **Download**: The script uses the `pytube` library to download a YouTube video specified by its URL. It provides callbacks to track the progress and completion of the download.

2. **Parse**: After downloading, it utilizes `ffmpeg-python` to extract frames from the video file. Frames are extracted at a rate of one frame per second and are saved to an output directory.

### Workflow Steps:
- The YouTube video URL is hard-coded within the script.
- The video is downloaded in MP4 format, tracked by progress callbacks.
- Upon download completion, a parsing function is automatically invoked.
- The parsing function uses `ffmpeg` to extract frames which are then saved to a predefined directory, with each file named according to the timestamp they were extracted plus a sequence number.

## recreate_with_ai.py

The `recreate_with_ai.py` script takes the frames extracted by the `download_and_parse.py` and transforms them with the help of a generative AI model, specifically Stable Diffusion Instruct Pix2Pix model from the `diffusers` library.

### Workflow Steps:
- The script checks for the presence of an `input_dir` with parsed frames.
- It creates an `output_dir` for the transformed frames if it doesn't exist.
- It loads a specified Stable Diffusion Instruct Pix2Pix model into the device (GPU or CPU).
- It iterates over each image in the `input_dir`, applies the AI model transformation based on a given prompt, and saves the output in the `output_dir`.
- Processing times are recorded and used to estimate remaining processing time, which is displayed after each frame is processed.

## Interaction between Scripts

1. **Video Download and Parsing**:
   - `download_and_parse.py` is executed first.
   - Once executed, it downloads a video and parses it into frames.

2. **Frame Transformation**:
   - After the frames are parsed, `recreate_with_ai.py` can be executed.
   - This script finds the images in the `input_dir`, which were the output of `download_and_parse.py`.
   - It goes through each frame, applies AI transformation, and saves the result in `output_dir`.

## Instructions for Use

1. Ensure that you have all required Python libraries installed (pytube, ffmpeg-python, PIL, torch, diffusers).
2. Run `download_and_parse.py` to fetch and parse the video into frames.
3. After the frames are extracted, run `recreate_with_ai.py` to process the images with the AI model.

## Important Notes

- The scripts are intended to be used in sequence, with `download_and_parse.py` first, followed by `recreate_with_ai.py`.
- The system must have sufficient resources to handle the processing load, especially for `recreate_with_ai.py`, which may require a GPU to function efficiently.
- The prompts and AI model configurations can be adjusted in `recreate_with_ai.py` to achieve different image transformation results.

By following these instructions and using the scripts, frames extracted from YouTube videos can be transformed into a curated set of images with AI-based stylization.