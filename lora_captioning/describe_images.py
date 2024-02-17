"""
/*2023Feb17 Notes*/

Project Objective:
- This script employs OpenAI's GPT-based vision model to generate detailed, lifelike captions for images of humans instead of BLIP or WD14 captioning services that leave a lot of detail out.

- The aim is to enhance the realism in AI-generated representations by focusing on nuances that define lifelike qualities: physical features, expressions, attire, and context.

Why We Caption Like This:
- Detailed Descriptions enhance the AI's understanding, enabling more realistic recreations.
- Emphasis on Artistic Elements (lighting, shading, color palette) helps in crafting more vibrant and contextually accurate renditions.
- Focusing on Physical and Emotional Details ensures each human representation carries individuality and depth.
- Incorporating Neutral and Artistic Terminology allows for inclusivity and precision, essential for realistic and respectful portrayals.

Key Features of the Captioning Approach:
1. **Detailing Physical Features:** Captions pinpoint unique characteristics for more personalized depictions.
2. **Expressions and Emotions:** Descriptions delve into the subject's emotional state, adding depth.
3. **Attire and Pose:** Detailing clothing and posture enriches the narrative and realism of each image.
4. **Contextualizing the Setting:** The backdrop is described to complement or contrast the subject, enhancing the overall realism.
5. **Neutral and Artistic Terminology:** Ensures descriptions are both inclusive and precise for accurate color and detail representation.

Output:
- The script outputs detailed captions that aim to inform the creation of lifelike AI-generated images. The goal is to closely replicate the nuances of human portraits as they might be described in detailed art and photography studies, using a methodology adapted for AI-driven applications.

Script’s Additional Feature:
- In addition to generating captions, this script seamlessly integrates with Google Drive to upload and manage image files, streamlining the process of selecting and processing images for captioning. This feature aids in efficient handling and organization of images to be described, facilitating a smoother workflow from image upload to caption generation.
"""

from upload_image_to_gdrive import upload_file
import os
import openai
from dotenv import load_dotenv
import requests
import logging
import json
import base64
import time

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Fetch the OpenAI API key
api_key = os.getenv('2023nov17_OPENAI_KEY')
if not api_key:
    raise EnvironmentError("Failed to retrieve 2023nov17_OPENAI_KEY from environment variables.")

# Set up the OpenAI client with your API key
openai.api_key = api_key

# Configuration for the GPT-4 model
VISION_MODEL = "gpt-4-vision-preview"
TEXT_PROMPT = """
If there is no human in the image, state there is nothing to describe. If there is a human, in no more than 150 words, describe in detail focusing on essential aspects for a lifelike representation:
- Key artistic elements like lighting, shading, and color palette.
- Critical facial features, expressions, and hair details.
- Clothing characteristics, fabric texture, and any visible movement or pose.
- The setting or background, and how it complements or contrasts with the person.
Please use objective and neutral art terms suitable for an artist to accurately recreate the scene.
"""
MAX_TOKENS = 150
USE_BASE64_ENCODING = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('describe_and_recreate')

def get_image_data(image_url, base64_encode=False):
    if base64_encode:
        response = requests.get(image_url)
        return base64.b64encode(response.content).decode('utf-8')
    else:
        return image_url

def describe_image(image_url, use_base64):
    image_data = get_image_data(image_url, base64_encode=use_base64)
    content_field = [
        {"type": "text", "text": TEXT_PROMPT}, 
        {"type": "image", "data" if use_base64 else "image_url": image_data}
    ]
   
    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": content_field
            }
        ],
        "max_tokens": MAX_TOKENS
    }
    
    print("Sending this to OpenAI: ")
    print(json.dumps(payload, indent=4))

    try:
        response = openai.ChatCompletion.create(**payload)
        
        print("Receiving this from OpenAI: ")
        print(json.dumps(response, indent=4))

        if response and 'choices' in response:
            message = response['choices'][0]['message']
            if message['role'] == 'assistant' and 'content' in message:
                if not message['content'].startswith("I'm sorry, but I am unable to view the image"):
                    return message['content']
        return None
    except openai.error.OpenAIError as e:
        logger.error("OpenAI API error occurred while describing image", exc_info=e)
        return None


def main():
    # Record the start time
    start_time = time.time()
    
    # Initialize a counter for the number of images processed
    images_processed = 0

    # Scan the current directory for .jpeg, .jpg, .png files
    current_dir = os.getcwd()
    image_files = [file for file in os.listdir(current_dir) if file.lower().endswith(('.jpeg', '.jpg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(current_dir, image_file)
        print(f"Processing {image_file}...")

        # Upload image file to Google Drive
        image_url = upload_file(image_path)  # Call the altered upload_file function with the image path

        # Describe the image if the upload was successful
        if image_url:
            description = describe_image(image_url, USE_BASE64_ENCODING)
            if description:
                print(f'Description of {image_file}:\n{description}')

                # Save the description to a text file with the same base name as the image file
                description_file_name = os.path.splitext(image_file)[0] + '.txt'
                description_path = os.path.join(current_dir, description_file_name)
                with open(description_path, 'w') as description_file:
                    description_file.write(description)
                print(f"Saved description to {description_file_name}")

                # After processing each image, increment the counter
                images_processed += 1
            else:
                print(f"Couldn't generate a valid description for {image_file}.")
        else:
            print(f"Failed to upload {image_file} to Google Drive.")

    # Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print the summary
    print(f"Total images processed: {images_processed}")
    print(f"Time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()