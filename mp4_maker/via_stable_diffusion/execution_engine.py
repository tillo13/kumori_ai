import os
import platform
import subprocess
import sys
from datetime import datetime
import time
from PIL import Image, ImageEnhance
import torch
from diffusers import DiffusionPipeline
import model_configs

from compel import Compel, ReturnedEmbeddingsType


import argparse

# Determine the OS and set cache path accordingly
if platform.system() == "Windows":
    # On Windows, use %LOCALAPPDATA% for cache directory
    HF_CACHE_HOME = os.path.expanduser(os.getenv("HF_HOME", os.path.join(os.getenv("LOCALAPPDATA"), "huggingface")))
else:
    # On Unix-like systems (Linux/macOS), default to XDG cache home or ~/.cache
    HF_CACHE_HOME = os.path.expanduser(os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface")))

DEFAULT_CACHE_PATH = os.path.join(HF_CACHE_HOME, "diffusers")
# Determine cache paths
DIFFUSERS_CACHE_PATH = os.path.join(HF_CACHE_HOME, "diffusers")
HUB_CACHE_PATH = os.path.join(HF_CACHE_HOME, "hub")

def get_user_selected_config():
    # This function is only called when no arguments are passed, i.e., when run interactively
    return list_models_and_choose()


def generate_with_long_prompt(pipe, cfg, device):
    prompt = cfg["PROMPT_TO_CREATE"]
    print(f"Processing long prompt of length {len(prompt)}")

    try:
        # Initialize compel with tokenizer and text_encoder
        compel = Compel(
            tokenizer=pipe.tokenizer, 
            text_encoder=pipe.text_encoder,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )
        print("Compel initialized with tokenizer and text encoder from the pipeline object.")

        # Generate both embeddings and pooled embeddings using Compel
        try:
            conditioning, pooled_conditioning = compel(prompt)
            images = pipe(prompt_embeds=conditioning, pooled_prompt_embeds=pooled_conditioning, num_inference_steps=cfg["NUM_INFERENCE_STEPS"]).images
            image = images[0]
            print("Image generated successfully with Compel.")
            return image
        except Exception as e:
            print(f"A non-critical error occurred while attempting to use Compel with pooled embeddings: {e}")
        
        # If the above fails, fallback to regular Compel (without pooled)
        compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
        conditioning = compel.build_conditioning_tensor(prompt)
        images = pipe(prompt_embeds=conditioning, num_inference_steps=cfg["NUM_INFERENCE_STEPS"]).images
        image = images[0]
        print("Image generated using fallback Compel method without pooled embeddings.")
        return image

    except Exception as e:
        # As a last resort, if compel still fails, use the default generation process without Compel
        print(f"Error during fallback Compel generation: {e}")
        try:
            images = pipe(prompt=prompt, num_inference_steps=cfg["NUM_INFERENCE_STEPS"]).images
            image = images[0]
            print("Image generated using default generation process due to exception.")
            return image
        except Exception as default_fallback_error:
            print(f"Error during default generation: {default_fallback_error}")
            return None  # Unable to generate image

def list_cached_models():
    print("Cached models:")

    # Check for diffusers cache
    list_cache(DIFFUSERS_CACHE_PATH, "Diffusers models")

    # Check for hub cache
    list_cache(HUB_CACHE_PATH, "Hub models")

def list_cache(cache_path, description):
    print(f"\n{description}:")
    if not os.path.isdir(cache_path):
        print(f"No cache directory found at '{cache_path}'.")
    else:
        model_directories = [d for d in os.listdir(cache_path) if os.path.isdir(os.path.join(cache_path, d))]
        if not model_directories:
            print("No cached models found.")
        else:
            for idx, model_dir in enumerate(model_directories, 1):
                model_dir_path = os.path.join(cache_path, model_dir)
                print(f"{idx}. {model_dir} (Location: {model_dir_path})")

def list_models_and_choose():
    global_settings = model_configs.GLOBAL_IMAGE_SETTINGS
    print("Global configurations:")
    print(f"   - Prompt: {global_settings['PROMPT_TO_CREATE']}")
    print(f"   - Number of images to create: {global_settings['NUMBER_OF_IMAGES_TO_CREATE']}")
    print(f"   - Inference steps: {global_settings['NUM_INFERENCE_STEPS']}")
    print()
    
    model_keys = list(model_configs.MODEL_CONFIGS.keys())
    print("Available models and their configurations:")
    for idx, model_name in enumerate(model_keys, 1):
        model_config = model_configs.MODEL_CONFIGS[model_name]
        print(f"{idx}. {model_name}")
        if 'MODEL_ID' in model_config:
            print(f"   - Model ID: {model_config['MODEL_ID']}")
        else:
            # For refiner models, print base and refiner model IDs
            print(f"   - Base Model ID: {model_config['MODEL_ID_BASE']}")
            print(f"   - Refiner Model ID: {model_config['MODEL_ID_REFINER']}")
        print()

    selected_config = None
    while selected_config is None:
        user_input = input("Select a model by number: ")
        try:
            model_idx = int(user_input) - 1  # Adjust for 0-based indexing
            if model_idx < 0 or model_idx >= len(model_keys):
                print("Invalid selection. Please try again.")
            else:
                selected_model_key = model_keys[model_idx]
                selected_config = model_configs.MODEL_CONFIGS[selected_model_key]
                print(f"You have selected: {selected_model_key}")

        except ValueError:
            print("Invalid input. Please enter a number.")
    
    return selected_config

def install_packages(packages):
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def format_time(seconds):
    return f"{int(seconds // 60)} minutes {seconds % 60:.2f} seconds"

def open_image(path):
    try:
        if sys.platform == "darwin":  # macOS
            subprocess.run(["open", path], check=True)
        elif sys.platform == "win32":  # Windows
            os.startfile(path)
        elif sys.platform.startswith('linux'):  # Linux
            subprocess.run(["xdg-open", path], check=True)
        else:
            print("Platform not supported for opening image.")
    except Exception as e:
        print(f"Failed to open image: {e}")

def post_process_image(image):
    config_values = model_configs.CURRENT_CONFIG
    factors = (config_values["UPSAMPLE_FACTOR"], config_values["SHARPNESS_ENHANCEMENT_FACTOR"],
               config_values["CONTRAST_ENHANCEMENT_FACTOR"])
    
    print("Resizing the image...")
    image = image.resize((image.width * factors[0], image.height * factors[0]), Image.LANCZOS)
    
    print("Enhancing image sharpness...")
    sharpness_enhancer = ImageEnhance.Sharpness(image)
    image = sharpness_enhancer.enhance(factors[1])
    
    print("Increasing image contrast...")
    contrast_enhancer = ImageEnhance.Contrast(image)
    image = contrast_enhancer.enhance(factors[2])
    
    print("Post-processing complete.")
    return image

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_device_name = torch.cuda.get_device_name(0)
        cuda_device_memory = torch.cuda.get_device_properties(0).total_memory
        cuda_device_memory_gb = cuda_device_memory / (1024 ** 3)
        hardware_summary = {
            "Device Type": "GPU",
            "Device Name": cuda_device_name,
            "Device Memory (GB)": f"{cuda_device_memory_gb:.2f}",
            "CUDA Version": torch.version.cuda,
        }
    else:
        device = torch.device("cpu")
        cpu_threads = torch.get_num_threads()
        hardware_summary = {
            "Device Type": "CPU",
            "Available Threads": cpu_threads,
        }

    # Print PyTorch version and device information
    print(f"PyTorch version: {torch.__version__}")
    device_info = f"Using device: {hardware_summary['Device Name']} with {hardware_summary['Device Memory (GB)']} GB of GPU memory and CUDA version {hardware_summary['CUDA Version']}" if "GPU" in hardware_summary["Device Type"] else f"Using device: CPU with {hardware_summary['Available Threads']} threads"
    print(device_info)

    return device, hardware_summary



def main(model_id, config_overrides=None, shared_timestamp=None):
    start_time = time.time()
    # Generate a timestamp to use in filenames/directories, preferring shared_timestamp if provided
    if shared_timestamp is None:
        # Use the current timestamp if a shared one isn't provided
        effective_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        # Use the shared timestamp if provided
        effective_timestamp = shared_timestamp



    # If config_overrides is None, it means we need to prompt the user for input
    if config_overrides is None:
        model_configs.CURRENT_CONFIG = get_user_selected_config()  # Use the prompt-based model selection
    else:

        # Update CURRENT_CONFIG with overrides
        for key, value in config_overrides.items():
            if key in model_configs.CURRENT_CONFIG:
                model_configs.CURRENT_CONFIG[key] = value

        # Update GLOBAL_IMAGE_SETTINGS and PILLOW_CONFIG with overrides if applicable
        global_settings = model_configs.GLOBAL_IMAGE_SETTINGS
        pillow_settings = model_configs.PILLOW_CONFIG

        for key, value in config_overrides.items():
            if key in global_settings:
                global_settings[key] = value
            elif key in pillow_settings:
                pillow_settings[key] = value

    # Install required Python packages for the chosen model
    install_packages(model_configs.REQUIRED_PACKAGES)

    # List any cached models if available
    list_cached_models()

    # Set up the device (GPU/CPU)
    device, hardware_summary = setup_device()

    # Now print out all the values that will be used to create the image
    print("\nConfiguration for image generation:")
    print(f"Model ID: {model_configs.CURRENT_CONFIG['MODEL_ID']}")
    print(f"Prompt: {model_configs.CURRENT_CONFIG['PROMPT_TO_CREATE']}")
    print(f"Number of Images: {model_configs.CURRENT_CONFIG['NUMBER_OF_IMAGES_TO_CREATE']}")
    print(f"Number of Inference Steps: {model_configs.CURRENT_CONFIG['NUM_INFERENCE_STEPS']}")
    print(f"Open Image After Creation: {model_configs.CURRENT_CONFIG['OPEN_IMAGE_AFTER_CREATION']}")
    print(f"Images Directory: {model_configs.CURRENT_CONFIG['IMAGES_DIRECTORY']}")
    print(f"Filename Template: {model_configs.CURRENT_CONFIG['FILENAME_TEMPLATE']}")
    print(f"Timestamp Format: {model_configs.CURRENT_CONFIG['TIMESTAMP_FORMAT']}")
    print(f"Add Safety Checker: {model_configs.CURRENT_CONFIG['ADD_SAFETY_CHECKER']}")
    print(f"Upsample Factor: {model_configs.CURRENT_CONFIG['UPSAMPLE_FACTOR']}")
    print(f"Sharpness Enhancement Factor: {model_configs.CURRENT_CONFIG['SHARPNESS_ENHANCEMENT_FACTOR']}")
    print(f"Contrast Enhancement Factor: {model_configs.CURRENT_CONFIG['CONTRAST_ENHANCEMENT_FACTOR']}")


    # Get the current configuration
    cfg = model_configs.CURRENT_CONFIG

    # Generate current timestamp
    current_timestamp = datetime.now().strftime(model_configs.GLOBAL_IMAGE_SETTINGS["TIMESTAMP_FORMAT"])
    

    # Update IMAGES_DIRECTORY based on the effective_timestamp, ensuring consistency or uniqueness as needed
    images_directory = f"{effective_timestamp}_{model_configs.CURRENT_CONFIG['IMAGES_DIRECTORY']}"
    os.makedirs(images_directory, exist_ok=True)  # Ensure this directory exists


    # Check if we're using a refiner model; this is specific to the SDXL Refiner model
    use_refiner = "MODEL_ID_REFINER" in cfg

    if use_refiner:
        # For SDXL Refiner, we have specific pipeline kwargs for base and refiner
        base_pipeline_kwargs = {
            "torch_dtype": torch.float16 if device.type == "cuda" else torch.float32,
            **cfg.get("ADDITIONAL_PIPELINE_ARGS_BASE", {})
        }
        refiner_pipeline_kwargs = {
            "torch_dtype": torch.float16 if device.type == "cuda" else torch.float32,
            **cfg.get("ADDITIONAL_PIPELINE_ARGS_REFINER", {})
        }
        # Load the base model pipeline
        base_model = DiffusionPipeline.from_pretrained(cfg["MODEL_ID_BASE"], **base_pipeline_kwargs).to(device)
        # Load the refiner model pipeline
        refiner_model = DiffusionPipeline.from_pretrained(cfg["MODEL_ID_REFINER"], **refiner_pipeline_kwargs).to(device)
        # Additional fraction used only for the SDXL Refiner model
        high_noise_fraction = cfg.get("HIGH_NOISE_FRACTION", 0.5)
    else:
        # For all other models, use a single pipeline with the following kwargs
        pipeline_kwargs = {
            "torch_dtype": torch.float16 if device.type == "cuda" else torch.float32,
            **cfg.get("ADDITIONAL_PIPELINE_ARGS", {}),
            "safety_checker": None if not cfg["ADD_SAFETY_CHECKER"] else None  # Set to None if safety_checker is False
        }
        # Load the pipeline
        pipe = DiffusionPipeline.from_pretrained(cfg["MODEL_ID"], **pipeline_kwargs).to(device)

    # Manage the SEED value; if it does not exist or is None, use the current time as the seed
    seed = model_configs.GLOBAL_IMAGE_SETTINGS.get("SEED") 
    if seed is None:
        seed = int(time.time())
        model_configs.GLOBAL_IMAGE_SETTINGS["SEED"] = seed  # Update the SEED in the global config
    
    torch.manual_seed(seed)
    generation_times = []

    for i in range(model_configs.GLOBAL_IMAGE_SETTINGS["NUMBER_OF_IMAGES_TO_CREATE"]):
        start_gen_time = time.time()
        print(f"Processing image {i+1} of {model_configs.GLOBAL_IMAGE_SETTINGS['NUMBER_OF_IMAGES_TO_CREATE']}...")

        # Retrieve model-specific additional arguments for the pipeline
        # These additional arguments will depend on whether we're using a refiner model or not
        additional_args = cfg.get("ADDITIONAL_PIPELINE_ARGS_BASE", {}) if use_refiner else cfg.get("ADDITIONAL_PIPELINE_ARGS", {})

        # Generate timestamp once for both filenames
        timestamp = datetime.now().strftime(model_configs.GLOBAL_IMAGE_SETTINGS["TIMESTAMP_FORMAT"])
        model_prefix = cfg.get("MODEL_PREFIX", "")
        
        # Create the base filename using the timestamp
        base_filename = model_configs.GLOBAL_IMAGE_SETTINGS["FILENAME_TEMPLATE"].format(
            model_prefix=model_prefix,
            timestamp=effective_timestamp
        )
        if use_refiner:
            # Generate with the base model
            image_latent = base_model(prompt=cfg["PROMPT_TO_CREATE"],
                                      num_inference_steps=int(model_configs.GLOBAL_IMAGE_SETTINGS["NUM_INFERENCE_STEPS"] * high_noise_fraction),
                                      denoising_end=high_noise_fraction,
                                      **additional_args).images[0]

            # Create a filename for initial images by appending "_initial" before the file extension
            initial_filename = base_filename.replace(".png", "_initial.png")
            initial_img_path = os.path.join(cfg["IMAGES_DIRECTORY"], initial_filename)
            
            # Save the initial image
            image_latent.save(initial_img_path)
            print(f"Initial image saved at: {initial_img_path}")
            
            # Process and save the final image with the refiner model:
            image = refiner_model(prompt=cfg["PROMPT_TO_CREATE"],
                                  num_inference_steps=model_configs.GLOBAL_IMAGE_SETTINGS["NUM_INFERENCE_STEPS"] - int(model_configs.GLOBAL_IMAGE_SETTINGS["NUM_INFERENCE_STEPS"] * high_noise_fraction),
                                  denoising_start=high_noise_fraction,
                                  image=image_latent).images[0]

        else:
            # Check for the existence of long prompt handling logic
            if len(cfg["PROMPT_TO_CREATE"]) > 77:
                print("Prompt exceeds token limit. Engaging long prompt handling function...")
                # Invoke the generate_with_long_prompt function
                image = generate_with_long_prompt(pipe, cfg, device)
            else:
                print("Prompt within token limit. Proceeding with regular generation process...")
                # Regular prompt processing logic
                image = pipe(prompt=cfg["PROMPT_TO_CREATE"],
                            num_inference_steps=model_configs.GLOBAL_IMAGE_SETTINGS["NUM_INFERENCE_STEPS"]).images[0]


        print("Starting post-processing with Pillow...")
        image = post_process_image(image)

        # Use the previously set `base_filename` as the final filename
        final_filename = base_filename  # This already has the correct format
        final_img_path = os.path.join(images_directory, final_filename)
        image.save(final_img_path)

        gen_time = time.time() - start_gen_time
        generation_times.append(gen_time)

        print(f"Image {i+1}/{cfg['NUMBER_OF_IMAGES_TO_CREATE']} saved in '{cfg['IMAGES_DIRECTORY']}' as {final_filename}")
        print(f"Full path: {os.path.abspath(final_img_path)}")
        print(f"Single image generation time: {format_time(gen_time)}")

        if model_configs.GLOBAL_IMAGE_SETTINGS["OPEN_IMAGE_AFTER_CREATION"]:
            open_image(final_img_path)


    total_time = time.time() - start_time
    avg_time = sum(generation_times) / len(generation_times) if generation_times else 0
    
    print("==== SUMMARY ====")
    print(f"Total execution time: {format_time(total_time)}")
    print(f"Average generation time per image: {format_time(avg_time)}")
    
    # Print out hardware summary information
    for key, val in hardware_summary.items():
        print(f"{key}: {val}")
    
    # Print out configuration values used
    print("\n--- Configuration Details ---")
    config_values = model_configs.CURRENT_CONFIG
    for key, val in config_values.items():
        print(f"{key}: {val}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image generation pipeline.")
    
    
    
    # Add only the model_id argument, the rest stays the same
    parser.add_argument('--model_id', type=str, default=model_configs.GLOBAL_IMAGE_SETTINGS["DEFAULT_MODEL_ID"], help='Model ID to automatically select for image generation.')

    parser.add_argument('--prompt', type=str, default=model_configs.GLOBAL_IMAGE_SETTINGS["PROMPT_TO_CREATE"], help='Custom prompt for image generation.')
    parser.add_argument('--num_images', type=int, default=model_configs.GLOBAL_IMAGE_SETTINGS["NUMBER_OF_IMAGES_TO_CREATE"], help='Number of images to create.')
    parser.add_argument('--num_steps', type=int, default=model_configs.GLOBAL_IMAGE_SETTINGS["NUM_INFERENCE_STEPS"], help='Number of inference steps.')
    parser.add_argument('--open_image', action='store_true', default=model_configs.GLOBAL_IMAGE_SETTINGS["OPEN_IMAGE_AFTER_CREATION"], help='Open image after creation.')
    parser.add_argument('--images_dir', type=str, default=model_configs.GLOBAL_IMAGE_SETTINGS["IMAGES_DIRECTORY"], help='Directory to save created images.')
    parser.add_argument('--filename_template', type=str, default=model_configs.GLOBAL_IMAGE_SETTINGS["FILENAME_TEMPLATE"], help='Template for naming image files.')
    parser.add_argument('--timestamp_format', type=str, default=model_configs.GLOBAL_IMAGE_SETTINGS["TIMESTAMP_FORMAT"], help='Format for timestamps in image filenames.')
    parser.add_argument('--add_safety_checker', action='store_true', default=model_configs.GLOBAL_IMAGE_SETTINGS["ADD_SAFETY_CHECKER"], help='Add a safety checker to the image generation pipeline.')

    # PILLOW_CONFIG arguments
    parser.add_argument('--upsample_factor', type=int, default=model_configs.PILLOW_CONFIG["UPSAMPLE_FACTOR"], help='Factor to upsample the image.')
    parser.add_argument('--sharpness_factor', type=float, default=model_configs.PILLOW_CONFIG["SHARPNESS_ENHANCEMENT_FACTOR"], help='Factor to enhance image sharpness.')
    parser.add_argument('--contrast_factor', type=float, default=model_configs.PILLOW_CONFIG["CONTRAST_ENHANCEMENT_FACTOR"], help='Factor to enhance image contrast.')

    #to pass in from storyline.json
    parser.add_argument('--shared_timestamp', type=str, help='Shared timestamp for consistent directory naming.')



    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    # Check if no arguments were passed
    if len(sys.argv) == 1:
        main(None)  # Running interactively, will prompt the user to choose a model
    else:
        # Find matching model configuration based on MODEL_ID
        # Loop through MODEL_CONFIGS and attempt to match the model_id
        selected_config = None
        for key, config in model_configs.MODEL_CONFIGS.items():
            if 'MODEL_ID' in config and config['MODEL_ID'] == args.model_id:
                selected_config = config
                break
            elif 'MODEL_ID_BASE' in config and config['MODEL_ID_BASE'] == args.model_id:
                selected_config = config
                break
            elif 'MODEL_ID_REFINER' in config and config['MODEL_ID_REFINER'] == args.model_id:
                selected_config = config
                break
        
        # Check if the selected_config is found
        if selected_config is None:
            print(f"Error: Model ID '{args.model_id}' not found in configurations. Exiting.")
            sys.exit(1)
        
        # Now selected_config contains the matched configuration
        model_configs.CURRENT_CONFIG = selected_config

        # Create dictionary of passed-in overrides
        config_overrides = {
            "PROMPT_TO_CREATE": args.prompt,
            "MODEL_ID": args.model_id,
            "NUMBER_OF_IMAGES_TO_CREATE": args.num_images,
            "NUM_INFERENCE_STEPS": args.num_steps,
            "OPEN_IMAGE_AFTER_CREATION": args.open_image,
            "IMAGES_DIRECTORY": args.images_dir,
            "FILENAME_TEMPLATE": args.filename_template,
            "TIMESTAMP_FORMAT": args.timestamp_format,
            "ADD_SAFETY_CHECKER": args.add_safety_checker,
            # PILLOW_CONFIG overrides
            "UPSAMPLE_FACTOR": args.upsample_factor,
            "SHARPNESS_ENHANCEMENT_FACTOR": args.sharpness_factor,
            "CONTRAST_ENHANCEMENT_FACTOR": args.contrast_factor,
        }

        # Adjust the call to the main function to include the shared_timestamp if provided
        if args.shared_timestamp:
            main(args.model_id, config_overrides, args.shared_timestamp)
        else:
            main(args.model_id, config_overrides)