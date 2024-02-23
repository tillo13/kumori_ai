import PIL
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
import glob
from datetime import datetime, timedelta
import os
import time
import numpy as np

# == GLOBAL SETTINGS ===
input_dir = "parsed_frames"
output_dir = "processed_frames"
prompt = "turn this photo into a Pixar movie"
model_id = "timbrooks/instruct-pix2pix"
device = "cuda" if torch.cuda.is_available() else "cpu"
num_inference_steps = 20
guidance_scale = 7
generator_seed = 1371
ESTIMATED_PROCESS_TIME = 60  # Estimated time per image in seconds 

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_images(input_dir, output_dir, prompt, model_id, device, num_inference_steps, guidance_scale, generator_seed):
    total_images = len(glob.glob(f"{input_dir}/*.png"))
    processing_times = []

    print("===BEGINNING SUMMARY===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Prompt: '{prompt}'")
    print(f"Model ID: {model_id}")
    print(f"Device: {device}")
    print(f"Number of inference steps: {num_inference_steps}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Generator seed: {generator_seed}")
    print(f"Total number of images to process: {total_images}")
    # We can't estimate time here yet; we'll do it after processing starts
    print("-" * 30)

    # Loading the InstructPix2Pix model
    print("Loading the InstructPix2Pix model...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None
    ).to(device)
    print("Model loaded successfully and moved to the selected device.")

    # Fetch image paths and sort them alphabetically
    image_paths = sorted(
    glob.glob(f"{input_dir}/*.png"),
    key=lambda path: path.lower()
)
    for idx, image_path in enumerate(image_paths):
            base_file_name = os.path.basename(image_path)
            print(f"Processing image {idx+1}/{total_images}: {base_file_name}")
            image = PIL.Image.open(image_path).convert("RGB")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{timestamp}_{idx+1:04d}.png"
            output_image_path = os.path.join(output_dir, output_filename)

            print("Starting image processing...")

            # Start time recording
            start_time = time.time()
            edited_images = pipe(prompt, image=image, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator_seed=generator_seed).images
            # End time recording
            end_time = time.time()

            # Calculate processing time
            time_taken = end_time - start_time
            processing_times.append(time_taken)

            # Save the processed image back to the output directory
            edited_images[0].save(output_image_path)
            print(f"Image processed in {time_taken:.2f} seconds.")
            print(f"Processed image saved as '{output_image_path}'.")

            # Estimate remaining time only after the first image has been processed
            if idx > 0 and processing_times:
                average_time = np.mean(processing_times)
                remaining_images = total_images - (idx + 1)
                remaining_time = average_time * remaining_images
                estimated_completion_time = datetime.now() + timedelta(seconds=remaining_time)
                time_remaining_str = str(timedelta(seconds=int(remaining_time)))  # Convert to HMS format

            # Calculate remaining time with an initial estimate or actual average
            remaining_images = total_images - (idx + 1)
            if processing_times:
                average_time = sum(processing_times) / len(processing_times)
            else:
                average_time = ESTIMATED_PROCESS_TIME  # Use initial estimated time

            remaining_time = average_time * remaining_images
            estimated_completion_time = datetime.now() + timedelta(seconds=remaining_time)
            time_remaining_str = str(timedelta(seconds=int(remaining_time)))  # Convert to HMS format

            print(f"Image processed in {time_taken:.2f} seconds.")
            print(f"Processed image saved as '{output_image_path}'.")

            print(f"Estimated time to completion: {estimated_completion_time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"({time_remaining_str})\n")

print("Starting to process images...")
process_images(input_dir, output_dir, prompt, model_id, device, num_inference_steps, guidance_scale, generator_seed)
print("Image processing completed.")