# YouTube Video Transcription and Summarization

This documentation describes the workflow and capabilities of `app.py`, a script that uses the `youtube-transcript-api` to fetch transcripts from a YouTube video, and `openai_utils.py`, which leverages OpenAI's GPT-3 model to generate summaries of those transcripts.

## app.py

The `app.py` script is designed to automate the process of extracting transcripts from YouTube videos and summarizing the content.

### Features:
- Extraction of the video ID from the YouTube URL.
- Retrieval of available transcripts for the video, prioritizing English (`en`).
- Collection of metadata regarding the transcript's characteristics, such as language specification and translatability.
- Resilience against failed transcript requests with a retry mechanism.
- Filtering transcript segments based upon timestamps.
- Integration with OpenAI API through `openai_utils.py` to generate a summary of the transcript.

### Workflow Steps:
1. **Video ID Extraction**: Determines the video ID from a given YouTube URL.
2. **Transcript Retrieval**: Fetches the transcript of the video in the designated language.
3. **Metadata Collection**: Gathers detailed information about the transcript and video traits.
4. **Error Handling**: Attempts to handle possible errors that may occur while fetching transcripts.
5. **Transcript Processing**: Saves the transcript to a file and prints it to the console along with metadata.
6. **Timestamp Analysis**: Optionally, the script can filter and print segments within a specified timestamp range.
7. **Summarization**: Passes the transcript to `openai_utils.py` for summarization, prints the summary, and measures the time taken.

## openai_utils.py

The `openai_utils.py` is a support module for `app.py`, providing the backbone for interaction with OpenAI services.

### Features:
- Initialization with OpenAI API key, loaded from an environment variable.
- A function for extracting a chat response from the model.
- Summarization capabilities utilizing GPT-3 or GPT-3.5-turbo models.

### Workflow Steps:
1. **API Key Validation**: Ensures that the OpenAI API key is set correctly before allowing the script to proceed.
2. **Model Interaction**: Defines functions to interact with different models for chat responses or summaries.
3. **Custom Prompts**: Crafts prompts and system messages catered to extracting summaries from YouTube transcripts.

## Interaction Between Scripts

- Upon executing `app.py`, the script uses the pre-defined URLs to attempt transcript retrieval.
- If successful, the extracted transcript, along with additional metadata, is passed to `openai_utils.py`.
- `openai_utils.py` invokes the OpenAI API to process the transcript and return a summary.
- The summary and associated processing duration are printed to the console by `app.py`.

## Instructions for Use

1. Confirm installation of the `youtube-transcript-api` and the `openai` Python packages.
2. Ensure that the environment variable for the OpenAI API key is set or loaded via `.env` with `dotenv`.
3. Modify the `URL_OF_YOUTUBE_VIDEO` in `app.py` to point to the desired YouTube video URL.
4. Run `app.py` to fetch the transcript and receive a summarized version.
5. Optionally, adjust the timestamp filtering in `app.py` to focus on specific portions of the transcript.

## Important Notes

- The script `app.py` should be used for videos with enabled transcription on YouTube.
- The summarized content's quality from `openai_utils.py` is contingent upon the clarity and completeness of the transcript.
- Both scripts must be present in the same directory for the summarization feature to function, as they are closely linked.
- The OpenAI model used for summarization may be specified or altered in `openai_utils.py`.

By adhering to these guidelines and utilizing the scripts provided, users can automate the transcription and summarization of YouTube videos, efficiently distilling content into key takeaways.