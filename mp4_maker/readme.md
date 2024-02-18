# MP4 Maker: Image & Video Generator

MP4 Maker is a comprehensive Python-based toolkit designed to generate captivating videos from textual descriptions using generative models like OpenAI's DALL·E for creating images and OpenAI's GPT models for generating captions. This project facilitates the seamless creation of video clips accompanied by audio, tailored captions, and visually stunning images generated based on user-defined prompts.

## Overview

The project consists of several Python scripts that work in tandem to:

- Archive existing images in the designated folder.
- Generate new images based on textual descriptions via OpenAI's DALL·E model.
- Download and trim audio tracks to fit the video length, fetching tracks from various online sources.
- Apply captions to images, adjusting for size, placement, and timing.
- Compile the captioned images and audio track into a final MP4 video file.
- Estimate the cost of API calls made during the image and captions generation process.

## Installation

To set up and run MP4 Maker on your system, follow these steps:

1. **Clone the repository:**
git clone 

2. **Install dependencies:**

Ensure Python 3.6 or later is installed. Then, install the required Python packages:
pip install -r requirements.txt

This command will install the necessary libraries, including `ffmpeg-python`, `mutagen`, `beautifulsoup4`, `requests`, `feedparser`, `pytube`, and `python-dotenv`, among others.

3. **Configure API keys and environment:**

- Obtain your OpenAI API key by signing up at [OpenAI](https://openai.com).
- Get API keys or access tokens for additional services if you plan to extend functionality (e.g., YouTube for audio sourcing).
- Create a `.env` file in the root directory of the project and add your OpenAI API key:

  ```
OPENAI_KEY="your_openai_api_key_here"
  ```

4. **Running the application:**

Launch the main script to start generating videos:
python mp4_maker_configs.py

## Components

### Image Generation

- **mp4_maker_configs.py** - Serves as the main script, orchestrating the video creation process by defining character and storyline descriptions, generating images via OpenAI's API, and calling the video assembly engine.

### Audio Handling

- **mp4_maker_fetch_music.py** - Facilitates downloading and trimming audio tracks to fit the generated video's length, with the flexibility to source music from various online platforms.

### Video Assembly

- **mp4_maker_engine.py** - Combines generated images, captions, and audio tracks to assemble the final video. Utilizes `ffmpeg` for video processing tasks.

### Utility Scripts

- **openai_utils.py** - Contains utility functions for interacting with OpenAI's APIs, including generating images and estimating costs. Also, includes chat completions functionality.
- **mp4_maker_random_rfm_selector.py** - Randomly selects RFM (Royalty Free Music) for the video background, providing a diverse selection of tracks.

## Features

- Text-to-image conversion using DALL·E for visually interpreting storylines.
- Automated video assembly with custom captions and audio tracks.
- Cost estimation for API usage, helping maintain budget awareness.
- Extensible design for incorporating additional data sources and generative models.

## Contributing

Contributions are welcome! If you have improvements or bug fixes, please open a pull request or issue.

## License

This project is licensed under [MIT License](LICENSE.txt). Feel free to use, modify, and distribute as per the license agreement.

## Acknowledgements

This project utilizes OpenAI's API; credits to OpenAI for providing access to powerful generative models. Additional thanks to various online platforms for music tracks used in this project.