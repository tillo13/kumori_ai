import os
import datetime
import shutil
import requests
from openai_utils import create_image
import mp4_maker_engine
import uuid

#====GLOBAL VARIABLES====#
CHARACTER_DESCRIPTION = """
The main characters of the story are a vicious white Alpha wolf known for his razor sharp teeth and a slim sleek body; a kind-hearted brown boston terrier who's the second in command; and a small, mistreated Beagle.
"""

STORYLINE_DESCRIPTION = """
The character is involved focuses on this exact activity: 
"""

GPT_IMAGE_DESCRIPTION = [
    "Illustrate the stark white alpha wolf with his razor sharp teeth, positioned as the intimidating leader of the pack.",
    "Show the brown boston terrier, exemplifying strength and kindness, acting as the second in command.",
    "Depict the Alpha wolf stealing a catch from the tiny Beagle during a river hunting.",
    "Visualize the boston terrier kindly giving his own catch to the Beagle, showing a stark contrast to the Alpha.",
    "Illustrate the moment another pack threatens their territory, with the Alpha wolf recklessly taking them on.",
    "Show the Alpha leaving his weaker pack mates to fend for themselves, exposing his cruel nature.",
    "Visualize the pack being attacked with the smaller Dachshund in striking danger.",
    "Show the boston terrier jumping in to save the Dachshund, demonstrating his bravery and loyalty.",
    "Illustrate the tension in the air as the boston terrier stands against the merciless Alpha, hinting at an upcoming showdown.",
    "Present the final stand between the Alpha and the boston terrier, leaving the fate of the pack hanging in the balance."
]

VIDEO_CAPTIONS = [
    "The intimidating white alpha leads with an iron paw...",
    "The kind-hearted boston terrier, stands as the pack's second in command...",
    "The alpha heartlessly steals the tiny Beagle's catch during a hunt...",
    "Showing compassion, the boston terrier gives his catch to the beaten Beagle...",
    "Their territory threatened, the Alpha wolf engages the enemy pack...",
    "Abandoned by the Alpha, the pack is left to fend for themselves against the invaders...",
    "In the chaos, a small Dachshund finds itself in peril...",
    "In the nick of time, the brave boston terrier steps in to protect the vulnerable Dachshund...",
    "Facing off against the cruel Alpha, the boston terrier takes a stand...",
    "In a final showdown, the pack's fate hangs in the balance..."
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

def archive_existing_images(base_directory):
    image_files = [f for f in os.listdir(base_directory) if f.endswith((".png", ".jpg", ".jpeg"))]
    
    if image_files:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_directory = os.path.join(base_directory, f"{timestamp}_images")
        os.makedirs(archive_directory, exist_ok=True)
        print(f"Archiving existing images to {archive_directory}")

        for filename in image_files:
            old_path = os.path.join(base_directory, filename)
            new_path = os.path.join(archive_directory, filename)
            shutil.move(old_path, new_path)
            print(f"Moved {filename} to archive.")
    else:
        print("No existing images to archive. The directory is clean.")


def get_image_files(folder_path):
    image_extensions = ('.jpg', '.png', '.jpeg')
    return sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)],
        key=lambda x: os.path.basename(x).lower()
    )

def main():
    working_directory = os.path.join(os.getcwd(), 'video_images')
    archive_existing_images(working_directory)
    os.makedirs(working_directory, exist_ok=True)

    # Generate images
    for i, image_description in enumerate(GPT_IMAGE_DESCRIPTION):
        # Add CHARACTER_DESCRIPTION and STORYLINE_DESCRIPTION to the prompt
        image_prompt = f"{CHARACTER_DESCRIPTION.strip()} {STORYLINE_DESCRIPTION.strip()} {image_description.strip()}"

        image_response = create_image(
            prompt=image_prompt,
            model=MODEL_NAME,
            n=1,
            quality=IMAGE_QUALITY,
            response_format="url",
            size=IMAGE_SIZE,
            style=IMAGE_STYLE,
            user_id=USER_ID
        )

        if 'data' in image_response:
            image_url = image_response['data'][0]['url']
            filename = f"image_{i:04d}.png"  # Ensure files are named sequentially
            download_image(image_url, working_directory, filename)

    # Verify that the number of generated images equals the number of descriptions
    generated_image_files = get_image_files(working_directory)
    if len(GPT_IMAGE_DESCRIPTION) != len(generated_image_files):
        print(f"The number of generated images ({len(generated_image_files)}) does not match the number of descriptions ({len(GPT_IMAGE_DESCRIPTION)}).")
        return  # Stop the execution if they don't match

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

from openai_utils import summarize_and_estimate_cost

summary_data_example = {
    'summary_text': 'Video generation completed.',
    'total_input_tokens': 0,  # Assuming no chat completions, you can fill this in if needed
    'total_output_tokens': 0,  # Assuming no chat completions, you can fill this in if needed
    'number_of_images': len(VIDEO_CAPTIONS)
}
summarize_and_estimate_cost(summary_data_example)

if __name__ == '__main__':
    main()