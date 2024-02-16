import os
import time
import datetime
import numpy as np
import cv2
import math
from PIL import Image
import torch
from diffusers import ControlNetModel
from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
import glob
import random
import csv
from style_template import styles

# Default Configuration variables
INPUT_FOLDER_NAME = 'sample_images'
OUTPUT_FOLDER_NAME = 'generated_images'
PROMPT = "a hyper-realistic full-color image of woman with blue eyes, extremely detailed facial features"
NEGATIVE_PROMPT = "(unrealistic, lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
IDENTITYNET_STRENGTH_RATIO = 1.0
ADAPTER_STRENGTH_RATIO = 1.0
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_SIDE = 1280
MIN_SIDE = 1024
NUMBER_OF_LOOPS = 5 
USE_RANDOM_STYLE = False


# Define the draw_keypoints function
def draw_keypoints(image_pil, keypoints, stickwidth=4, colors=None):
    if colors is None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    limb_sequence = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    keypoints = np.array(keypoints)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limb_sequence)):
        index = limb_sequence[i]
        color = colors[index[0]]

        x = keypoints[index][:, 0]
        y = keypoints[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))),
                                   (int(length / 2), stickwidth),
                                   int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img, polygon, color)

    out_img = (out_img * 0.6).astype(np.uint8)  # Apply transparency by blending with the original image

    for idx_kp, kp in enumerate(keypoints):
        color = colors[idx_kp % len(colors)]
        x, y = kp
        out_img = cv2.circle(out_img, (int(x), int(y)), stickwidth, color, -1)

    out_img_pil = Image.fromarray(out_img)
    return out_img_pil

def load_instant_id_model():
    # Path to InstantID models
    face_adapter = './checkpoints/ip-adapter.bin'
    controlnet_path = './checkpoints/ControlNetModel'

    # Load the face encoder "antelopev2"
    app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Load ControlNetModel from checkpoints
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    print("Face encoder and ControlNetModel have been loaded, starting generation...")

    # Load the InstantID pipeline with the pretrained base model and control net
    base_model_path = 'stabilityai/stable-diffusion-xl-base-1.0'
    try:
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=torch.float16
        )
        if torch.cuda.is_available():
            print("It appears torch.CUDA is available. Using it!")
            pipe.cuda()
        
        # Use the debug statements here to inspect the pipe object if needed
        #print(pipe)
        #print(dir(pipe))
        
        pipe.load_ip_adapter_instantid(face_adapter)
    except AttributeError as e:
        print("Failed to load pipeline or call load_ip_adapter_instantid:")
        print(e)
        # If we're here, pipe is not instantiated correctly
        pipe = None  # or raise an exception to fail loudly

    return pipe, app

def resize_image(image, max_side=1280, min_side=1024, pad_to_max_side=False):
    w, h = image.size
    ratio = float(min_side) / min(h, w)
    w, h = int(ratio * w), int(ratio * h)
    if max(w, h) > max_side:
        ratio = float(max_side) / max(h, w)
        w, h = int(ratio * w), int(ratio * h)
    image = image.resize((w, h), Image.BILINEAR)
    
    if pad_to_max_side:
        new_image = Image.new("RGB", (max_side, max_side))
        new_image.paste(image, ((max_side - w) // 2, (max_side - h) // 2))
        return new_image

    return image

def generate_image(pipe, app, input_image_path, folder_name, identitynet_strength_ratio, adapter_strength_ratio, num_inference_steps, guidance_scale, seed, selected_style_prompt, selected_negative_prompt):
    # Extract just the filename from the input path
    filename_only = os.path.basename(input_image_path)
    print(f"Starting image generation process for {filename_only}...")

    # Load the image file
    face_image = Image.open(input_image_path)
    
    # Resize and optionally pad the image to match the model's expected input dimensions
    face_image = resize_image(face_image, max_side=MAX_SIDE, min_side=MIN_SIDE, pad_to_max_side=True)

    print(f"Image {filename_only} has been resized for processing.")

    # Convert PIL image to a numpy array for face detection
    face_image_np = np.array(face_image)
    # Get face information with embeddings and keypoints
    face_info = app.get(face_image_np)[-1]
    print("Face information extracted from the image.")

    face_emb = face_info['embedding']
    face_kps = draw_keypoints(face_image, face_info['kps'])

    # Set the random seed
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    # Generate a new image with the chosen seed
    new_images = pipe(
        prompt=PROMPT,
        negative_prompt=selected_negative_prompt,
        image_embeds=face_emb,
        image=face_kps,
        controlnet_conditioning_scale=identitynet_strength_ratio,
        ip_adapter_scale=adapter_strength_ratio,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images

    # Get the first (and only) image in the .images list
    new_image = new_images[0]
    print("Image has been successfully generated, proceeding with saving.")

    # Save the generated image with a timestamp and filename
    new_file_name, output_image_path = save_image_with_timestamp(new_image, input_image_path, folder_name)
    
    # Return the new filename and the image
    return new_file_name, new_image


def save_image_with_timestamp(image, input_image_path, folder_name):
    print("Saving the generated image with a timestamp...")
    # Extract the base filename without the extension
    base_filename = os.path.splitext(os.path.basename(input_image_path))[0]
    # Extract the file extension and remove the dot
    file_extension = os.path.splitext(input_image_path)[1][1:].lower()
    # Create a timestamp in the specified format
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # Combine the base filename, original extension, and timestamp for the output filename
    output_filename = f"{base_filename}_{file_extension}_{timestamp}.png"
    # Create the full output path
    output_image_path = os.path.join(folder_name, output_filename)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Save the image
    image.save(output_image_path)
    print(f"Generated image saved to '{output_image_path}'")

    # Return the new filename and full path for logging
    return output_filename, output_image_path


def process_image(input_image_path, output_folder, pipe, face_analyzer, loop):
    # Generate a random seed
    seed = random.randint(0, 2**32 - 1)

    # Select a random style if USE_RANDOM_STYLE is True
    selected_style_name = "(No style)"
    selected_style_prompt = PROMPT
    selected_negative_prompt = NEGATIVE_PROMPT
    if USE_RANDOM_STYLE:
        selected_style_name, (selected_style_prompt, selected_negative_prompt) = random.choice(list(styles.items()))
        selected_style_prompt = selected_style_prompt.format(prompt=PROMPT)
        selected_negative_prompt = f"{NEGATIVE_PROMPT}, {selected_negative_prompt}"

    # Randomize settings for this particular image
    identitynet_strength_ratio = random.uniform(0.8, 1.5)
    adapter_strength_ratio = random.uniform(0.3, 1.0)
    num_inference_steps = random.randint(20, 60)
    guidance_scale = random.uniform(5.0, 15.0)
    
    # Print settings for verification
    print(f"Settings for {os.path.basename(input_image_path)}:\n"
          f"- IdentityNet strength ratio: {identitynet_strength_ratio}\n"
          f"- Adapter strength ratio: {adapter_strength_ratio}\n"
          f"- Number of inference steps: {num_inference_steps}\n"
          f"- Guidance scale: {guidance_scale}\n")
    
    # Call the generate_image function with the randomized settings
    success = True
    error_message = ''

    # Call the generate_image function within a try block and log the information to CSV
    try:    
        new_file_name, new_image = generate_image(pipe, face_analyzer, input_image_path, OUTPUT_FOLDER_NAME, identitynet_strength_ratio, adapter_strength_ratio, num_inference_steps, guidance_scale, seed, selected_style_prompt, selected_negative_prompt)

        # Log the successful generation outcome
        log_to_csv('image_generation_log.csv', os.path.basename(input_image_path), new_file_name, identitynet_strength_ratio, adapter_strength_ratio, num_inference_steps, guidance_scale, seed, True, '', selected_style_name, selected_style_prompt, selected_negative_prompt)

    except Exception as e:
        error_message = str(e)
        print(f"Error processing {input_image_path}: {e}")
        
        # Log the error
        log_to_csv('image_generation_log.csv', os.path.basename(input_image_path), '', identitynet_strength_ratio, adapter_strength_ratio, num_inference_steps, guidance_scale, seed, False, error_message, selected_style_name)


def log_to_csv(logfile_path, image_name, new_file_name, identitynet_strength_ratio, adapter_strength_ratio, num_inference_steps, guidance_scale, seed, success=True, error_message='', style_name="", selected_style_prompt="", selected_negative_prompt=""):
    # Check if the CSV file already exists
    file_exists = os.path.isfile(logfile_path)
    with open(logfile_path, 'a', newline='') as csvfile:
        fieldnames = ['image_name', 'new_file_name', 'identitynet_strength_ratio', 'adapter_strength_ratio', 'num_inference_steps', 'guidance_scale', 'seed', 'success', 'error_message', 'style_name', 'selected_style_prompt', 'selected_negative_prompt']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If the file doesn't exist, we need to write the header
        if not file_exists:
            writer.writeheader()

        # Write the log data
        writer.writerow({
            'image_name': image_name,
            'new_file_name': new_file_name,
            'identitynet_strength_ratio': identitynet_strength_ratio,
            'adapter_strength_ratio': adapter_strength_ratio,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
            'success': success,
            'error_message': error_message,
            'style_name': style_name,
            'selected_style_prompt': selected_style_prompt,
            'selected_negative_prompt': selected_negative_prompt
        })

def main():
    start_time = time.time()
    total_processing_time = 0.0  # Initialize outside the loop to accumulate time across all loops

    # Load the models needed for image generation
    pipe, face_analyzer = load_instant_id_model()

    # List all image files in the `INPUT_FOLDER_NAME`
    image_files = glob.glob(INPUT_FOLDER_NAME + '/*.png') + \
                  glob.glob(INPUT_FOLDER_NAME + '/*.jpg') + \
                  glob.glob(INPUT_FOLDER_NAME + '/*.jpeg')
    
    # Shuffle the image files randomly
    random.shuffle(image_files)

    total_files = len(image_files)
    print(f"Found {total_files} images in {INPUT_FOLDER_NAME}. Preparing to process...")

    # Initial assumption for processing time per image (in seconds)
    default_processing_time = 30 * 60  # 30 minutes per image

    # Estimate the total time before starting the first image
    estimated_total_time = default_processing_time * total_files
    print(f"Estimated total time: {estimated_total_time / 60:.2f} minutes")

    # Process each image file for the designated number of loops
    for loop in range(NUMBER_OF_LOOPS):
        print(f"\nStarting loop {loop + 1} of {NUMBER_OF_LOOPS}...")
        loop_start_time = time.time()  # Start time for the current loop
        
        for idx, input_image_path in enumerate(image_files):
            file_start_time = time.time()
            print(f"Processing in loop {loop + 1}, image {idx + 1}/{total_files}: {input_image_path}")

            try:
                process_image(input_image_path, OUTPUT_FOLDER_NAME, pipe, face_analyzer, loop + 1)
                file_processing_time = time.time() - file_start_time
                loop_processing_time = time.time() - loop_start_time
                total_processing_time += file_processing_time
                average_processing_time = total_processing_time / ((loop * total_files) + idx + 1)

                images_processed_in_current_loop = idx + 1
                images_remaining_in_current_loop = total_files - images_processed_in_current_loop
                total_images_remaining = images_remaining_in_current_loop + ((NUMBER_OF_LOOPS - loop - 1) * total_files)
                
                # This is the estimated time for the remaining images in the current loop and subsequent loops
                estimated_remaining_time = average_processing_time * total_images_remaining

                # Output detailed time tracking information
                print(f"\n{total_images_remaining} total images remaining.")
                print(f"Last file took {file_processing_time:.2f} seconds, or {file_processing_time / 60:.2f} minutes.")
                print(f"All files processed so far average: {average_processing_time:.2f} seconds, or {average_processing_time / 60:.2f} minutes.")
                print(f"Estimated total time left for all loops: {estimated_remaining_time / 60:.2f} minutes.\n")

            except Exception as e:
                print(f"Error processing {input_image_path} in loop {loop + 1}: {e}")

        loop_end_time = time.time()
        # Loop processing summary
        print(f"Total time for loop {loop + 1}: {(loop_end_time - loop_start_time) / 60:.2f} minutes.")

    # Calculate the total time taken and print summary
    total_elapsed_time = time.time() - start_time
    print("\n===FINAL SUMMARY===")
    print(f"Total loops completed: {NUMBER_OF_LOOPS}")
    print(f"Total images processed per loop: {total_files}")
    print(f"Overall total time: {total_elapsed_time / 60:.2f} minutes")

if __name__ == "__main__":
    main()