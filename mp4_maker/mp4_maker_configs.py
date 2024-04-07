import os
import datetime
import shutil
import requests
from openai_utils import create_image, summarize_and_estimate_cost
import mp4_maker_engine
import uuid
import time

RETRY_NUMBER = 5
RETRY_OPENAI_CALL = 3
SLEEP_BETWEEN_RETRIES = 2  
NUMBER_OF_VIDEOS = 3  


#====GLOBAL VARIABLES====#
CHARACTER_DESCRIPTION = """
The main character of the story is a young Asian girl in her teens with black hair and brown eyes.
"""


STORYLINE_DESCRIPTION = """
Make the storyline about this exact event with the main character:
"""

GPT_IMAGE_DESCRIPTION = [
    "Illustrate the main character watering plants in her garden.",
    "Show a small droplet of water getting absorbed by a plant sprout and turning it into a large, blooming flower instantly.",
    "Depict her getting embraced by the flower’s petaled arms and getting showered in pollen.",
    "Illustrate her transformation into a superhero clad in a green dress with leafy patterns, and her new identity as 'Blossom Bud'.",
    "Showcase Blossom Bud standing on top of a skyscraper, as she brings about a burst of flora in the urban jungle.",
    "Depict Blossom Bud creating a rainforest out of a barren desert, demonstrating her power on an immense scale.",
    "Illustrate Blossom Bud standing on a cliff overlooking the sea, about to transform a lifeless ocean into a vibrant underwater forest.",
    "Finally, depict Blossom Bud in space watching Earth turn lush and verdant, her astonishing rejuvenation of life and growth reaching cosmic proportions."
]

VIDEO_CAPTIONS = [
    "A teenage girl is watering plants in her garden.",
    "She notices with a single small drop of water, a sprout can bloom into a large flower.",
    "The flower engulfs her and showers her with pollen.",
    "Emerging from the flower, she now dons a green dress with leafy patterns, transforming into Blossom Bud.",
    "Discovering her new abilities, Blossom Bud brings a burst of flora to a city from atop a skyscraper.",
    "Blossom Bud turns a barren desert into a lush rainforest, showcasing her immense power.",
    "Blossom Bud stands on a cliff overlooking the sea, ready to make the lifeless ocean a vibrant underwater forest.",
    "Eventually, Blossom Bud stands in space watching Earth turn lush and verdant, her powers rejuvenating life on a cosmic scale."
]

IMAGE_SIZE = "1024x1024"
MODEL_NAME = "dall-e-3"
IMAGE_QUALITY = "hd"
IMAGE_STYLE = "vivid"
USER_ID = str(uuid.uuid4())  # Generate a random UUID4 as the unique user ID


VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1080
FONT_SIZE = 36
FONT_COLOR = 'white'
CAPTION_OFFSET_Y = '0.10*h'
DISPLAY_DURATION_PER_IMAGE = 4
AUDIO_TRACK_TYPE = 'halloween'
OUTPUT_FILENAME_PATTERN = 'output_with_captions'

def download_image(url, dest_folder, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(dest_folder, filename), 'wb') as f:
            f.write(response.content)
    else:
        print(f"Error downloading {url}: Status code {response.status_code}")

def archive_existing_assets(base_directory):
    # You can specify other extensions if your videos have a different one
    image_extensions = ('.jpg', '.png', '.jpeg')
    video_extensions = ('.mp4',)
    
    # Get all asset files
    asset_files = [f for f in os.listdir(base_directory) if f.endswith(image_extensions + video_extensions)]
    
    if asset_files:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_directory = os.path.join(os.path.dirname(base_directory), "archive", f"{timestamp}_assets")
        os.makedirs(archive_directory, exist_ok=True)
        
        print(f"Archiving existing assets ({len(asset_files)}) to {archive_directory}")

        for filename in asset_files:
            old_path = os.path.join(base_directory, filename)
            new_path = os.path.join(archive_directory, filename)
            shutil.move(old_path, new_path)
            print(f"Moved {filename} to archive.")

        print(f"Number of files archived this time: {len(asset_files)}")
        print(f"Total files in archive directory: {len(os.listdir(archive_directory))}")

        return len(asset_files), len(os.listdir(archive_directory))

    else:
        print("No existing assets to archive. The directory is clean.")
        return 0, 0


def get_image_files(folder_path):
    image_extensions = ('.jpg', '.png', '.jpeg')
    return sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)],
        key=lambda x: os.path.basename(x).lower()
    )

def create_image_with_retry(prompt, model, n, quality, response_format, 
                            size, style, user_id, retry_limit):
    for i in range(retry_limit):
        try:
            response = create_image(
                prompt=prompt,
                model=model,
                n=n,
                quality=quality,
                response_format=response_format,
                size=size,
                style=style,
                user_id=user_id
            )
            if 'data' in response:
                return response
        except Exception as e:
            if i < retry_limit - 1:
                print(f"Error occurred while creating image: {e}. Retrying...")
                time.sleep(SLEEP_BETWEEN_RETRIES)
                continue
            else:
                print(f"Error occurred while creating image: {e}. All retries failed.")
                raise
    return None

def main():
    working_directory = os.path.join(os.getcwd(), 'latest_video_assets')
    os.makedirs(working_directory, exist_ok=True)
    
    # Call archive function before generating new assets
    archive_existing_assets(working_directory)
    
    success = False
    
    for j in range(RETRY_NUMBER):  # Introducing retry mechanism
        try:
            # Generate images
            for i, image_description in enumerate(GPT_IMAGE_DESCRIPTION):
                image_prompt = f"{CHARACTER_DESCRIPTION.strip()} {STORYLINE_DESCRIPTION.strip()} {image_description.strip()}"

                image_response = create_image_with_retry(
                    prompt=image_prompt,
                    model=MODEL_NAME,
                    n=1,
                    quality=IMAGE_QUALITY,
                    response_format="url",
                    size=IMAGE_SIZE,
                    style=IMAGE_STYLE,
                    user_id=USER_ID,
                    retry_limit=RETRY_OPENAI_CALL
                )

                if image_response is not None:
                    image_url = image_response['data'][0]['url']
                    filename = f"image_{i:04d}.png"  # Ensure files are named sequentially
                    download_image(image_url, working_directory, filename)

            # Verify that the number of generated images equals the number of descriptions
            generated_image_files = get_image_files(working_directory)
            if len(GPT_IMAGE_DESCRIPTION) != len(generated_image_files):
                print(f"Attempt {j+1} failed: The number of generated images ({len(generated_image_files)}) does not match the number of descriptions ({len(GPT_IMAGE_DESCRIPTION)}). Retrying...")
                continue  # if the number of images doesn't match, then it will just skip to the next loop iteration, rather than exiting the loop completely

            print(f"Successfully generated all images in attempt {j+1}.")
            success = True
            break  # This will end the loop only when we have a successful generation of the images

        except Exception as e:  # Catch all exceptions and print the error message
            print(f"Attempt {j+1} failed due to error: {e}. Retrying...")
            time.sleep(SLEEP_BETWEEN_RETRIES)  # Sleep before next attempt
    
    if not success:
        # If all attempts failed, log the failure and return
        print(f"All {RETRY_NUMBER} attempts failed.")
        return 0
    
    # Video generation starts here:
    caption_properties = {
        'font_size': FONT_SIZE,
        'font_color': FONT_COLOR,
        'caption_offset_y': CAPTION_OFFSET_Y,
    }

    mp4_maker_engine.main(
        VIDEO_CAPTIONS,
        working_directory,
        VIDEO_WIDTH,
        VIDEO_HEIGHT,
        caption_properties,
        DISPLAY_DURATION_PER_IMAGE,
        AUDIO_TRACK_TYPE,
        OUTPUT_FILENAME_PATTERN
    )

    # After generating the video, archive the new assets
    archive_existing_assets(working_directory)
    
    return 1  # If it reaches here means video generation was successful

if __name__ == "__main__":
    start_time = time.time()

    # Configurable parameters
    COST_PER_IMAGE_IN_HD = 0.080  # Cost per image in HD from DALL-E
    COST_PER_TOKEN_FOR_INPUT = 0.01 / 1000  # Cost per token for 'gpt-4-0125-preview'
    TOKENS_PER_IMAGE = 765  # Estimated number of tokens consumed per image for a 1024 * 1024 image with "high" detail level
    NUMBER_OF_VIDEOS = 3  # Number of videos to generate

    # Derived parameters
    number_of_images = len(GPT_IMAGE_DESCRIPTION)  # Number of images based on the number of descriptions
    total_successes = sum(main() for _ in range(NUMBER_OF_VIDEOS))

    # Cost estimation
    image_generation_cost = COST_PER_IMAGE_IN_HD * number_of_images * NUMBER_OF_VIDEOS
    cost_per_image_for_description = (TOKENS_PER_IMAGE / 1000) * COST_PER_TOKEN_FOR_INPUT
    image_description_cost = cost_per_image_for_description * number_of_images * NUMBER_OF_VIDEOS

    # Time calculations
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Summary printout
    print(f"====SUMMARY====")
    print(f"Total time taken: {elapsed_time} seconds")
    print(f"Total videos created: {total_successes}")
    print(f"Total images created: {total_successes * len(GPT_IMAGE_DESCRIPTION)}")  # Correct this line if VIDEO_CAPTIONS was meant instead
    print(f"Estimated cost of image generation: ${image_generation_cost:.2f}")
    print(f"Estimated cost of image description: ${image_description_cost:.2f}")

    # Additional potential summary
    summary_data_example = {
        'summary_text': 'Video generation completed.',
        'total_input_tokens': total_successes * TOKENS_PER_IMAGE,  # This might represent the tokens consumed for the prompts
        'total_output_tokens': 0,  # It's not clear how output tokens are used in your case
        'number_of_images': total_successes * len(GPT_IMAGE_DESCRIPTION),  # Total number of images created
        'estimated_cost': image_generation_cost + image_description_cost
    }
    summarized_cost = summarize_and_estimate_cost(summary_data_example)
    print(f"Summarized cost: ${summarized_cost:.2f}")