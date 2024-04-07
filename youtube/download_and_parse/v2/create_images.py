import os
import glob
from datetime import datetime, timedelta
import time
import numpy as np
import PIL
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline

# === GLOBAL SETTINGS ===
DIRECTORIES = sorted(filter(os.path.isdir, os.listdir('.')), key=os.path.getmtime)
if DIRECTORIES:
    INPUT_DIR = DIRECTORIES[-1]  # Use the last modified directory
else:
    raise Exception("No directories found for input frames.")

OUTPUT_DIR = f"{INPUT_DIR}_processed"
MOVIE_STYLE = "japanese anime studio ghibli"
PROMPT = f"turn this photo into an overly exaggerated {MOVIE_STYLE} movie"
MODEL_ID = "timbrooks/instruct-pix2pix"
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7
GENERATOR_SEED = 1313

# Check if the output directory exists and is not empty
if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
    raise Exception(f"The directory '{OUTPUT_DIR}' is not empty. Please check it before proceeding.")

# If the output directory is empty, delete it
elif os.path.exists(OUTPUT_DIR):
    os.rmdir(OUTPUT_DIR)

# Create the output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Determine whether to use a GPU or CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_id):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None
        ).to(device)
        print(f"Model loaded successfully on {device.upper()}.")  # Prints whether GPU (CUDA) or CPU is used
    except Exception as e:
        device = "cpu"
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            safety_checker=None
        ).to(device)
        print(f"Failed to load the model on CUDA. Fallback: Model loaded successfully on {device.upper()}.")
    return pipe, device

def process_images(input_dir, output_dir, prompt, model_id, num_inference_steps, guidance_scale, generator_seed):    
    total_start_time = time.time()
    total_images = len(glob.glob(f"{input_dir}/*.png"))
    processing_times = []

    # Load the model with either GPU or CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe, used_device = load_model(model_id)

    print(f"Using {used_device.upper()} for inference.")
    print("===BEGINNING SUMMARY===")
    print(f"Are we using NVIDIA/CUDA: {torch.cuda.is_available()}")
    print(f"Input directory: {input_dir}")  # Use parameter names, not global variable names
    print(f"Output directory: {output_dir}")
    print(f"Prompt: {prompt}")
    print(f"Model ID: {model_id}")
    print(f"Device: {device}")
    print(f"Number of inference steps: {num_inference_steps}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Generator seed: {generator_seed}")
    print(f"Total number of images to process: {total_images}")
    print("-" * 30)

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
        output_image_path = os.path.join(OUTPUT_DIR, output_filename)

        print("Starting image processing...")
        start_time = time.time()

        edited_images = pipe(PROMPT, image=image, num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE, generator_seed=GENERATOR_SEED).images
        
        end_time = time.time()
        time_taken = end_time - start_time
        processing_times.append(time_taken)
        edited_images[0].save(output_image_path)
        
        average_time = np.mean(processing_times)
        remaining_images = total_images - (idx + 1)
        remaining_time = average_time * remaining_images
        estimated_completion_time = datetime.now() + timedelta(seconds=remaining_time)
        time_remaining_str = str(timedelta(seconds=int(remaining_time)))  # Convert to HMS format

        print(f"Image processed in {time_taken:.2f} seconds.")
        print(f"Processed image saved as '{output_image_path}'.")
        print(f"Estimated time to completion: {estimated_completion_time.strftime('%Y-%m-%d %H:%M:%S')} ({time_remaining_str})\n")

    total_end_time = time.time()
    total_time_taken = total_end_time - total_start_time
    total_time_str = str(timedelta(seconds=int(total_time_taken)))  # Convert to HMS format

    print("===SUMMARY===")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Prompt: '{PROMPT}'")
    print(f"Model ID: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print(f"Number of inference steps: {NUM_INFERENCE_STEPS}")
    print(f"Guidance scale: {GUIDANCE_SCALE}")
    print(f"Generator seed: {GENERATOR_SEED}")
    print(f"Total number of images processed: {total_images}")
    print(f"Total time taken for processing: {total_time_str}")
    print("-" * 30)

# No DEVICE argument passed to process_images
try:
    print("Starting to process images...")
    process_images(INPUT_DIR, OUTPUT_DIR, PROMPT, MODEL_ID, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, GENERATOR_SEED)
    print("Image processing completed.")
except Exception as e:
    # This will print the exception message you raised in the process_images function
    print("Error:", str(e))