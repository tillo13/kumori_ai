import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

#====GLOBAL SETTINGS====
MODEL_ID = "timbrooks/instruct-pix2pix"
LOCAL_IMAGE_PATH = "img_2045.png"  # Replace with your image path
PROMPT = "remove hat"
NUM_INFERENCE_STEPS = 10
IMAGE_GUIDANCE_SCALE = 1

# Function to clear GPU memory if CUDA is available.
def attempt_clear_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Check if CUDA is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
attempt_clear_cuda_memory()  # Attempt to clear any unused memory from GPU
print(f"Using device: {DEVICE}")

#====END OF GLOBAL SETTINGS====

# Function to load and prepare a local image
def load_local_image(file_path):
    with Image.open(file_path) as image:
        image = ImageOps.exif_transpose(image)  # Correct the orientation
        image = image.convert("RGB")  # Convert image to RGB
    return image

# Function to generate an image using Stable Diffusion InstructPix2Pix model
def generate_image(pipe, prompt, input_image, steps, scale):
    try:
        result = pipe(prompt=prompt, image=input_image, num_inference_steps=steps, image_guidance_scale=scale)
        return result.images[0]
    except torch.cuda.CudaOutOfMemoryError:
        print("CUDA out of memory. Attempting to clear cache and retry on CPU.")
        attempt_clear_cuda_memory()
        DEVICE = "cpu"  # Switch to CPU
        return generate_image(pipe.to(DEVICE), prompt, input_image, steps, scale)

# Load the model, specify the torch_dtype based on the available device, and set the scheduler
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    safety_checker=None
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to(DEVICE)

# Load the local image
input_image = load_local_image(LOCAL_IMAGE_PATH)

# Generate the image using the prompt and input image
output_image = generate_image(pipe, PROMPT, input_image, NUM_INFERENCE_STEPS, IMAGE_GUIDANCE_SCALE)

# Display the resulting image
output_image.show()

# Optionally, save the output image to a file
output_image.save("output_image.jpg")