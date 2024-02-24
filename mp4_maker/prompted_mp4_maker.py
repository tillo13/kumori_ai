import os
import datetime
import shutil
import requests
from openai_utils import create_image, summarize_and_estimate_cost
import mp4_maker_engine
import uuid
import time
import webbrowser
import sys

RETRY_NUMBER = 5
RETRY_OPENAI_CALL = 3
SLEEP_BETWEEN_RETRIES = 2  
NUMBER_OF_VIDEOS = 1


#====GLOBAL VARIABLES====#
CHARACTER_DESCRIPTION = """
The main character of the story is an ordinary man from india.
"""

STORYLINE_DESCRIPTION = """
Make the storyline about this exact event with the main character:
"""

GPT_IMAGE_DESCRIPTION = [
    "Illustrate a man casually discovering and swallowing an odd pill.",
    "Show the man realizing he can produce a perfect clone of himself.",
    "Depict the clone, identical to the original, heading to work.",
    "Show the original man, deeply relaxed, engrossed in his video game.",
    "Illustrate the man understanding the true potential of his newfound ability.",
    "Visualize a slew of duplicates, emerging as political leaders among people.",
    "Depict the clones standing proud as they assume presidency position.",
    "Show the clones playing a crucial role in dangerous and groundbreaking scientific experimentation.",
    "Illustrate the revolutionary space travel discovery made possible by their tests.",
    "Visualize the man and his clones, conquering Mars, Jupiter and other planets.",
    "Depict them ruling over the entire solar system, their influence unparalleled.",
    "Illustrate them expanding their dominion beyond our solar system, taking control of the Milky Way.",
    "Show the man, now as a god, standing in the midst of an infinite sea of clones."
]

VIDEO_CAPTIONS = [
    "Unveiling the man, who stumbles upon and down an enigmatic pill.",
    "Witness an unusual discovery - his ability to clone himself.",
    "Observe as the clone regularly attends work, indistinguishable from his original.",
    "See the man spending his day gaming, while his clone works.",
    "Gradually, he comprehends the true power his newfound ability possesses.",
    "With multiple clones, he emerges as a political leader.",
    "Watch as they ascend to power, becoming presidents.",
    "Now assisting in hazardous chemical experiments, they make new scientific breakthroughs.",
    "Their scientific advancements revolutionize space travel.",
    "They conquer planets like Mars and Jupiter with their space ships.",
    "The solar system is under their rein, with their unparalleled powers.",
    "Their rule extends to the Milky Way, their influence unmatched.",
    "As the sole inhabitants, he and his clones have ascended to godhood."
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

def initial_prompt():
    while True:
        user_response = input("Ok, if you already have images, we can skip this.  Do you want to create new images? 1 for yes, 2 for no: ").strip()
        if user_response == "1":
            return True
        elif user_response == "2":
            expected_number_of_images = len(GPT_IMAGE_DESCRIPTION)
            while True:
                user_confirm = input(f"Please place {expected_number_of_images} images in the directory.\n"
                                     f"Did you add files to the directory? 1 for yes, 2 for no: ").strip()
                if user_confirm == "1":
                    return False
                elif user_confirm == "2":
                    print("Please add the required images before continuing.")
                else:
                    print("Invalid choice. Please enter 1 or 2.")
        else:
            print("Invalid choice. Please enter 1 or 2.")


def user_prompt(image_number):
    while True:
        print(f"\nImage {image_number} has been created. What would you like to do?")
        print("1: Continue with the next image")
        print("2: Retry creating the same image")
        print("3: Change the image description")
        user_choice = input("Enter your choice (1, 2, or 3): ").strip()
        
        if user_choice == "1":
            return "continue"
        elif user_choice == "2":
            return "retry"
        elif user_choice == "3":
            return "change"
        else:
            print("Invalid choice. Please enter 1, 2, or 3.\n")

def download_image(url, dest_folder, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(dest_folder, filename), 'wb') as f:
            f.write(response.content)
        # After saving the image locally, open the image URL in Chrome
        webbrowser.get('chrome').open(url)  # This line will open the URL in Chrome
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

    create_images = initial_prompt()
    success = False  # Assume failure unless proven otherwise

    if create_images:
        archive_existing_assets(working_directory)
        i = 0
        while i < len(GPT_IMAGE_DESCRIPTION):
            success = False
            for j in range(RETRY_NUMBER):
                try:
                    image_description = GPT_IMAGE_DESCRIPTION[i]
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
                        filename = f"image_{i:04d}.png"
                        download_image(image_url, working_directory, filename)

                        # Success, image created and downloaded
                        success = True
                        
                        action = user_prompt(i + 1)
                        if action == "retry":
                            continue  # This will cause the loop to retry the same image
                        elif action == "change":
                            new_description = input(f"Enter new description for image {i + 1}: ")
                            GPT_IMAGE_DESCRIPTION[i] = new_description.strip()
                            continue  # Retry the same image with new description
                        else:
                            i += 1  # Move on to the next image
                        
                        break  # Successfully created an image, break out of the retry loop

                    else:
                        raise Exception(f"Failed to generate image {i + 1}")

                except Exception as e:
                    print(f"Attempt {j + 1} failed due to error: {e}. Retrying...")
                    time.sleep(SLEEP_BETWEEN_RETRIES)
                    continue

            if not success:
                print("Failed to create all images after retries.")
                return 0  # Exit function due to failed image creation

    else:
        generated_image_files = get_image_files(working_directory)
        if len(generated_image_files) != len(GPT_IMAGE_DESCRIPTION):
            print(f"Error: The number of images in the directory ({len(generated_image_files)}) "
                  f"does not match the expected number ({len(GPT_IMAGE_DESCRIPTION)}).")
            return 0  # Exit function due to image file count mismatch

    # Proceed to video generation since image requirements were met
    caption_properties = {
        'font_size': FONT_SIZE,
        'font_color': FONT_COLOR,
        'caption_offset_y': CAPTION_OFFSET_Y,
    }

    while True:
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

        user_audio_choice = input("Do you want to keep this audio? 1 for Yes, 2 for No: ").strip()
        if user_audio_choice == "1":
            break  # User is satisfied with the audio, break the loop
        elif user_audio_choice == "2":
            print("Fetching a different audio track...")
        else:
            print("Invalid input. Please enter 1 for Yes or 2 for No.")
    
    archive_existing_assets(working_directory)
    return 1  # If we reached this point, the function executed successfully


if __name__ == "__main__":
    start_time = time.time()

    # Configurable parameters
    COST_PER_IMAGE_IN_HD = 0.080  # Cost per image in HD from DALL-E
    COST_PER_TOKEN_FOR_INPUT = 0.01 / 1000  # Cost per token for 'gpt-4-0125-preview'
    TOKENS_PER_IMAGE = 765  # Estimated number of tokens consumed per image for a 1024 * 1024 image with "high" detail level

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

    # delete audio folder
    audio_folder = './audios'
    print(f"Deleting audio folder with youtube audio in it...")
    if os.path.exists(audio_folder):
        shutil.rmtree(audio_folder)

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
        'total_input_tokens': total_successes * TOKENS_PER_IMAGE,  # tokens consumed for the prompts
        'total_output_tokens': 0,  # Not used in this example
        'number_of_images': total_successes * len(GPT_IMAGE_DESCRIPTION),  # total number of images created
        'estimated_cost': None  # Leave it as None or remove 'estimated_cost' from this dictionary
    }
    summarize_and_estimate_cost(summary_data_example)  # Will still do its work but not print any cost estimation

    # Calculate and print the cost separately
    total_estimated_cost = image_generation_cost + image_description_cost
    print(f"Total estimated cost: ${total_estimated_cost:.2f}")