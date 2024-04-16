import os
import random
import csv
import gc
import glob
from datetime import datetime
import time
from pathlib import Path
from style_template import style_list
from PIL import Image, ImageOps

# Default Configuration variables
LOGGING_ENABLED = True
RANDOM_MODEL_ENABLED = True #will default to last one in def choose_random_model list if False

# Define the global variable for the percentage chance of choosing a style and boolean to decide
PERC_OF_STYLE = 20
USE_RANDOM_STYLE = True

INPUT_FOLDER_NAME = 'sample_images'
OUTPUT_FOLDER_NAME = 'generated_images'
LOG_FILENAME = 'generation_log.csv'

INPUT_FOLDER_NAME = 'sample_images'
OUTPUT_FOLDER_NAME = 'generated_images'
LOG_FILENAME = 'generation_log.csv'
logfile_path = os.path.join(os.getcwd(), LOG_FILENAME)
MAX_SIDE = 1280  
MIN_SIDE = 1024  
NUMBER_OF_LOOPS = 50 


PROMPT= "completely bald man, greying beard, confidence, ultra detail, hyper realistic, relaxed, charisma, expertise, wisdom, friendliness, candid laughter, serious concentration, charismatic gaze, blue eyes, realistic skin, high-definition 4K, ARRI ALEXA 65, subject focus, bokeh background, business, professional, realism"

NEGATIVE_PROMPT = "balding, full head of hair, text, words, hat, baseball hat, cap, headwear, grainy, overexposed, blurry, abstract, glitchy, low resolution, low quality, watermarks, text, deformed, mutated, cross-eyed, disfigured, unprofessional, ill-lit, unhealthy tones, bad posture, uniform lighting, dull, lifeless, cross-eyed, ugly, disfigured, saturated, oversaturated."

# IDENTITYNET_STRENGTH_RATIO_RANGE: This range is critical for capturing and preserving the unique semantic facial information, such as eye color or nose shape. Based on InstantID's emphasis on zero-shot, high-fidelity identity preservation with single images, we choose this higher range to ensure the generated image retains the distinct characteristics of the individual's identity as closely as possible. This aligns with InstantID's capability of achieving high fidelity without the need for extensive fine-tuning or multiple reference images like LoRA.
IDENTITYNET_STRENGTH_RATIO_RANGE = (1.35, 2.2)

# ADAPTER_STRENGTH_RATIO_RANGE: This range influences the extent to which the model captures and replicates the intricate details from the reference facial image, thanks to the IP-Adapter's role in encoding these details and offering spatial control. A balanced approach aims to enhance detail fidelity while avoiding excessive saturation, drawing on InstantID's strength in seamlessly blending styles and capturing personal identity features.
ADAPTER_STRENGTH_RATIO_RANGE = (0.55, 1.2)

# NUM_INFERENCE_STEPS_RANGE: A higher number of steps allows for a more detailed, refined image generation process, aligning with InstantID's approach to generating personalized, high-quality images efficiently. Given the emphasis on accuracy and detail over speed, this range ensures the model has ample opportunity to process and incorporate the nuances of the individual's identity, as well as intricate style details.
NUM_INFERENCE_STEPS_RANGE = (20, 100)  

# GUIDANCE_SCALE_RANGE: This parameter is fine-tuned to ensure a strong alignment of the generated image with textual prompts while preserving the unique identity attributes captured by IdentityNet and ControlNet. The chosen range reflects InstantID's capability for detailed, faithful replication of identity attributes within various stylistic interpretations, ensuring that the final image is not only stylistically coherent but also an accurate reflection of the individual’s identity.
GUIDANCE_SCALE_RANGE = (3.0, 10.0)    

# IDENTITYNET_STRENGTH_RATIO_RANGE = (1.35, 1.5)
# ADAPTER_STRENGTH_RATIO_RANGE = (0.55, 0.75)  # reduced to minimize oversaturation
# NUM_INFERENCE_STEPS_RANGE = (1, 2)  
# GUIDANCE_SCALE_RANGE = (3.0, 10.0)    


# Dynamically create the STYLES list from imported style_list
STYLES = [style["name"] for style in style_list]

def choose_random_style():
    # Generate a random number between 1 and 100
    chance = random.randint(1, 100)

    # Check if the random number falls within the percentage chance of choosing a style
    if chance <= PERC_OF_STYLE:
        chosen_style = random.choice(STYLES)
        print(f"PERC_OF_STYLE ({PERC_OF_STYLE}%) chance hit with {chance}: Choosing a random style -> {chosen_style}")
        return chosen_style
    else:
        print(f"PERC_OF_STYLE ({PERC_OF_STYLE}%) chance missed with {chance}: Using 'no style'")
        return "no style"


def choose_random_model():
    models = [
        'stablediffusionapi/omnigenxl-nsfw-sfw',
        'stablediffusionapi/realism-engine-sdxl-v30',
        'wangqixun/YamerMIX_v8',
        'RunDiffusion/Juggernaut-XL-Lightning',
        'RunDiffusion/juggernaut-xl-v8',
]
    if RANDOM_MODEL_ENABLED:
        return random.choice(models)
    else:
        return models[-1]

HUGGINGFACE_MODEL = choose_random_model()

def get_modified_model_name(model_name):
    # Check and modify only the specific model name
    if model_name == 'stablediffusionapi/omnigenxl-nsfw-sfw':
        return 'stablediffusionapi/omnigenxl'  # Return the modified name
    return model_name  # Return the original name for all other models

def get_random_image_file(input_folder):
    valid_extensions = [".jpg", ".jpeg", ".png"]
    files = [file for file in Path(input_folder).glob("*") if file.suffix.lower() in valid_extensions]
    if not files:
        raise FileNotFoundError(f"No images found in directory {input_folder}")
    return str(random.choice(files))

def resize_and_pad_image(image_path, max_side, min_side, pad_color=(255, 255, 255)):
    # Open an image using PIL
    image = Image.open(image_path)

    # Calculate the scale and new size
    ratio = min(min_side / min(image.size), max_side / max(image.size))
    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))

    # Resize the image
    image = image.resize(new_size, Image.BILINEAR)
    
    # Calculate padding
    delta_w = max_side - new_size[0]
    delta_h = max_side - new_size[1]

    # Pad the resized image to make it square
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    image = ImageOps.expand(image, padding, pad_color)

    return image

def log_to_csv(logfile_path, image_name, new_file_name='Unknown', identitynet_strength_ratio=0.0, adapter_strength_ratio=0.0, num_inference_steps=0, guidance_scale=0.0, seed=0, success=True, error_message='', style_name="", prompt="", negative_prompt="", time_taken=0.0, current_timestamp="", huggingface_model=""):
    os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
    file_exists = os.path.isfile(logfile_path)

    with open(logfile_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_name', 'new_file_name', 'identitynet_strength_ratio', 'adapter_strength_ratio', 'num_inference_steps', 'guidance_scale', 'seed', 'success', 'error_message', 'style_name', 'prompt', 'negative_prompt', 'time_taken', 'current_timestamp', 'huggingface_model']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
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
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'time_taken': time_taken,
            'current_timestamp': current_timestamp,
            'huggingface_model': huggingface_model  # Ensure this matches with your fieldnames
        })

def initial_image(generate_image_func):
    overall_start_time = time.time()
    total_time_taken = 0.0 

    # Initialize a counter for processed images at the beginning of the function
    processed_images_count = 0

    # List all image files in the `INPUT_FOLDER_NAME`
    image_files = glob.glob(f'{INPUT_FOLDER_NAME}/*.png') + \
                  glob.glob(f'{INPUT_FOLDER_NAME}/*.jpg') + \
                  glob.glob(f'{INPUT_FOLDER_NAME}/*.jpeg')
    
    # Check if we found any images
    if not image_files:
        raise FileNotFoundError(f"No images found in directory {INPUT_FOLDER_NAME}")
    
    # Print the count of detected image files
    print(f"Processing a total of {len(image_files)} image(s) in '{INPUT_FOLDER_NAME}'")

    # Shuffle the image files randomly
    random.shuffle(image_files)
    
    total_images = len(image_files)  # Get the total number of images to process

    def get_local_timestamp():
        return datetime.now().strftime("%Y%b%d @ %I:%M%p local time")    

    for loop in range(NUMBER_OF_LOOPS):
        global HUGGINGFACE_MODEL  # Inform Python that you're using the global variable
        HUGGINGFACE_MODEL = choose_random_model()  # Choose a new model for each loop iteration
        print(f"in loop, selected Huggingface model for loop {loop+1}: {HUGGINGFACE_MODEL}")
        print(f"Starting loop {loop+1} of {NUMBER_OF_LOOPS} at {get_local_timestamp()}")
        print(f"Logging enabled: {LOGGING_ENABLED}")

        for image_number, face_image_path in enumerate(image_files, start=1):
            
            loop_start_time = datetime.now()
            face_image = [face_image_path]
            basename = os.path.basename(face_image_path)
            processed_images_count += 1

            # Resize and pad the image before processing
            processed_image = resize_and_pad_image(
                image_path=face_image_path,
                max_side=MAX_SIDE,
                min_side=MIN_SIDE
            )

            if USE_RANDOM_STYLE:
                style_name = choose_random_style()
            else:
                style_name = "(No style)"
            
            # Print out the chosen style here
            print(f"Chosen style for this iteration: {style_name}")

            identitynet_strength_ratio = random.uniform(*IDENTITYNET_STRENGTH_RATIO_RANGE)
            adapter_strength_ratio = random.uniform(*ADAPTER_STRENGTH_RATIO_RANGE)
            num_inference_steps = random.randint(*NUM_INFERENCE_STEPS_RANGE)
            guidance_scale = random.uniform(*GUIDANCE_SCALE_RANGE)
            seed = random.randint(0, 2**32 - 1)
            
            # Print settings for the current image BEFORE processing it
            print_generation_settings(basename, style_name, identitynet_strength_ratio, 
                                      adapter_strength_ratio, num_inference_steps, guidance_scale, seed,
                                      image_number, total_images,HUGGINGFACE_MODEL)

            # Here, the generate_image_func is supposedly called and image processing happens
            _, _, generated_file_paths = generate_image_func(
                face_image=face_image,
                pose_image=None,
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                style_name=style_name,
                enhance_face_region=True,
                num_steps=num_inference_steps,
                identitynet_strength_ratio=identitynet_strength_ratio,
                adapter_strength_ratio=adapter_strength_ratio,
                guidance_scale=guidance_scale,
                seed=seed
            )
            
            loop_end_time = datetime.now()
            loop_time_taken = (loop_end_time - loop_start_time).total_seconds()

            # Immediately print the time taken and current time.
            print(f"Time taken to process image: {loop_time_taken:.2f} seconds")

            # Update the total time taken with this image's processing time
            total_time_taken += loop_time_taken

            # Calculate the average time taken per image
            average_time_per_image = total_time_taken / image_number

            current_timestamp = loop_end_time.strftime("%Y-%m-%d %H:%M:%S")  # Current time after processing
            print(f"Current timestamp: {current_timestamp}")

            # Calculate estimated remaining time considering the images left in this loop and the additional loops
            remaining_images_this_loop = total_images - image_number
            remaining_images_in_additional_loops = (NUMBER_OF_LOOPS - (loop + 1)) * total_images
            total_remaining_images = remaining_images_this_loop + remaining_images_in_additional_loops
            estimated_time_remaining = average_time_per_image * total_remaining_images

            # Display the estimated time remaining including remaining loops
            print(f"Estimated time remaining (including loops): {estimated_time_remaining // 60:.0f} minutes, {estimated_time_remaining % 60:.0f} seconds")

            # Display the overall average time per image in seconds
            print(f"Overall average time per image: {average_time_per_image:.2f} seconds")

            # Display the total number of remaining images to process including looping
            print(f"Total remaining images to process (including loops): {total_remaining_images}")
            print(f"Loop completed at {get_local_timestamp()}")

            success = True  # Assuming generation was successful.
            error_message = ""  # Assuming no error.

            # Loop through each generated file (assuming generate_image_func returns generated file paths for this demonstration)
            for generated_file_path in generated_file_paths:
                input_file_name_without_extension = os.path.splitext(basename)[0]
                current_timestamp_formatted = datetime.now().strftime("%Y%m%d%H%M%S")

                # Handling specific model name modification requirement for
                if HUGGINGFACE_MODEL == 'stablediffusionapi/omnigenxl-nsfw-sfw':
                    model_name_safe = 'omnigenxl'
                else:
                    model_name_safe = HUGGINGFACE_MODEL.replace("/", "_").replace("\\", "_")

                # Just before the renaming/moving file operation
                model_name_safe = get_modified_model_name(HUGGINGFACE_MODEL).replace("/", "_").replace("\\", "_")
                new_file_name = f"{model_name_safe}_{input_file_name_without_extension}_{current_timestamp_formatted}.png"
                new_file_path = os.path.join(OUTPUT_FOLDER_NAME, new_file_name)

                # Ensuring the output directory exists
                os.makedirs(OUTPUT_FOLDER_NAME, exist_ok=True)

                # Rename (move) the file
                try:
                    os.rename(generated_file_path, new_file_path)
                    print(f"Image saved as {new_file_path}")
                except Exception as e:
                    print(f"Error during file renaming: {e}")

                # Now check if logging is enabled and log the operation
                if LOGGING_ENABLED:
                    modified_huggingface_model = get_modified_model_name(HUGGINGFACE_MODEL)
                    log_to_csv(logfile_path, basename, new_file_name, identitynet_strength_ratio, adapter_strength_ratio, num_inference_steps, guidance_scale, seed, True, "", style_name, PROMPT, NEGATIVE_PROMPT, loop_time_taken, current_timestamp, modified_huggingface_model)

            del generated_file_paths  # Explicitly delete large variables
            gc.collect()  # Call garbage collection

    # At the end of the initial_image() function, add:
    total_elapsed_time = time.time() - overall_start_time
    print("\n===FINAL SUMMARY===")
    print(f"Total loops completed: {NUMBER_OF_LOOPS}")
    print(f"Total images processed per loop: {len(image_files)}")
    print(f"Overall total images processed: {NUMBER_OF_LOOPS * len(image_files)}") # Multiplied by the number of loops
    print(f"Overall total time: {total_elapsed_time / 60:.2f} minutes")

def print_generation_settings(basename, style_name, identitynet_strength_ratio, adapter_strength_ratio, num_inference_steps, guidance_scale, seed, image_number, total_images, HUGGINGFACE_MODEL):  

    print("===IMAGE GENERATION DATA SUMMARY===")    
    # Existing print statements follow
    print(f"- Image {image_number} of {total_images}\n"
          f"- Filename: {basename}\n"
          f"- Use random style: {USE_RANDOM_STYLE}\n"
          f"- Style: {style_name}\n"
          f"- IdentityNet strength ratio: {identitynet_strength_ratio:0.2f}\n"
          f"- Adapter strength ratio: {adapter_strength_ratio:0.2f}\n"
          f"- Number of inference steps: {num_inference_steps}\n"
          f"- Guidance scale: {guidance_scale:0.2f}\n"
          f"- Seed: {seed}\n"
          f"- Input folder name: {INPUT_FOLDER_NAME}\n"
          f"- Output folder name: {OUTPUT_FOLDER_NAME}\n"
          f"- Prompt: {PROMPT}\n"
          f"- Negative prompt: {NEGATIVE_PROMPT}\n"
          f"- Number of loops: {NUMBER_OF_LOOPS}\n"
          f"- HuggingFace Model: {HUGGINGFACE_MODEL}\n")
    
    print("===DEFINING COMPLETE, GENERATING IMAGE...===")