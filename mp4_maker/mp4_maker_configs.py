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
The main character of the story is a muscular, blonde-haired, blue eye man with strength and extraordinary physical capabilities. 
"""

STORYLINE_DESCRIPTION = """
The photo realistic hyper detailed character is involved on this exact activity: 
"""

GPT_IMAGE_DESCRIPTION = [
    "Illustrate the man in a striking stance, displaying his strong build.",
    "Visualize the man working out at the gym with intense dedication.",
    "Show the man deadlifting 300 pounds.",
    "As he loses excess weight and starts building muscle, visualize the man growing stronger.",
    "Illustrate him as he squat-lifts an astonishing load of 500 pounds.",
    "Depict the man bench pressing 1000 pounds, shocking the onlookers.",
    "Visualize him trying to lift the entire heavyweight of the gym building.",
    "Illustrate him pulling a massive cargo ship through the ocean",
    "Depict the man exerting his force to bring landscapes together.",
    "Show the man's hair turning blue like a blazing fire as he bench presses the planet earth.",
    "Illustrate the man in highly defined neon blue and black colors, with blue fire-like hair, lifting planets on a stake.",
    "Visualize the man smashing galaxies together for amusement, as if playing with toys.",
    "Present the  man evolving into a divine entity who holds the universe in his hands."
]

VIDEO_CAPTIONS = [
    "Meet the 6ft buff guy with blonde hair, exuding power and dominance.",
    "Here he is at the gym, pushing his limits and working out.",
    "He lifts with ease, deadlifting 300 pounds!",
    "He looks stronger in each frame, his muscles getting more pronounced.",
    "Now, see him squat 500 pounds, smashing personal bests!",
    "He moves onto bench presses, shockingly lifting 1000 pounds!",
    "Watch as he elevates his work out, lifting the entire gym building.",
    "His strength knows no bounds, as he pulls a cargo ship through the ocean.",
    "Not content, he brings continents closer together with his sheer force.",
    "Presenting a dramatic change, his hair becomes blue and fiery as he bench-presses our planet Earth.",
    "His features glow neon blue and black, and his fiery hair stands out as he lifts staked planets.",
    "For a bit of fun, he flings galaxies together, showing off his incomparable strength.",
    "Witness the true form of a god as he holds the universe in his hands."
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
            else:
                print(f"Successfully generated all images in attempt {j+1}.")
                success = True
                break  # Success, break out of the retry loop

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


summary_data_example = {
    'summary_text': 'Video generation completed.',
    'total_input_tokens': 0,  # Assuming no chat completions, you can fill this in if needed
    'total_output_tokens': 0,  # Assuming no chat completions, you can fill this in if needed
    'number_of_images': len(VIDEO_CAPTIONS)
}
summarize_and_estimate_cost(summary_data_example)

if __name__ == "__main__":
    total_successes = sum(main() for _ in range(NUMBER_OF_VIDEOS))

    print(f"====SUMMARY====")
    print(f"Total videos created: {total_successes}")
    print(f"Total images created: {total_successes * len(VIDEO_CAPTIONS)}")