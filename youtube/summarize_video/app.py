#2023-Aug-25 
#testing github copilot on a use case: https://en.wikipedia.org/wiki/GitHub_Copilot
#using https://pypi.org/project/youtube-transcript-api/
#filename: ingest_youtube.py
#github https://github.com/tillo13/youtube_transcribe_openai_summarize


from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, YouTubeRequestFailed
import time
from urllib.parse import urlparse, parse_qs
from openai_utils import get_summary_from_transcript
import os

#enter a url of the youtube video you want to transcribe 
URL_OF_YOUTUBE_VIDEO = "https://www.youtube.com/watch?v=iK4u95thQn0" # has transcripts in song
URL_OF_YOUTUBE_VIDEO = "https://www.youtube.com/watch?v=9KI1gSTDCCg" # has transcripts in story
#URL_OF_YOUTUBE_VIDEO = "https://www.youtube.com/watch?v=51TpMIEyUxk" ##no transcripts test

def get_video_id(video_url):
    parsed_url = urlparse(video_url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            query_params = parse_qs(parsed_url.query)
            return query_params['v'][0]
        if parsed_url.path[:7] == '/embed/':
            return parsed_url.path.split('/')[2]
        if parsed_url.path[:3] == '/v/':
            return parsed_url.path.split('/')[2]
    return None

def get_transcript_and_metadata(video_url, language_codes=['en']):
    video_id = get_video_id(video_url)

    if video_id is None:
        raise Exception(f"Could not parse video URL: {video_url}")
    
    attempts = 0
    while attempts < 3:  # you may adjust the number of attempts
        try:
            transcripts = YouTubeTranscriptApi.get_transcript(video_id)
            break
        except TranscriptsDisabled:
            print(f"Transcription is disabled for the video: {video_url}")
            print("Please choose another video where transcription is enabled.")
            return None, None
        except YouTubeRequestFailed:
            attempts += 1
            print(f"Request to YouTube failed. Attempt: {attempts}. Retrying...")
            time.sleep(5)  # wait for 5 seconds before retrying

    if attempts == 3:  # all attempts failed
        print("Could not retrieve the transcript. Please try again later.")
        return None, None

    try:
        transcripts = YouTubeTranscriptApi.get_transcript(video_id)
    except TranscriptsDisabled:
        print(f"Transcription is disabled for the video: {video_url}")
        print("Please choose another video where transcription is enabled.")
        return None, None

    transcribed_text = " ".join([caption['text'] for caption in transcripts])

    # Getting the list of all available transcripts
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    for transcript in transcript_list:
        is_generated = transcript.is_generated
        is_translatable = transcript.is_translatable

    translatable_to_spanish = "es" in [lang['language_code'] for lang in transcript_list.find_transcript(language_codes).translation_languages]
    translatable_to_japanese = "ja" in [lang['language_code'] for lang in transcript_list.find_transcript(language_codes).translation_languages]

    metadata = {
            'video_id': video_id,
            'language': transcript_list.find_transcript(language_codes).language,
            'language_code': transcript_list.find_transcript(language_codes).language_code,
            'is_generated': is_generated,
            'is_translatable': is_translatable,
            'translatable_to_spanish': translatable_to_spanish,
            'translatable_to_japanese': translatable_to_japanese,
            'transcripts': transcripts  # added this line
        }

    return transcribed_text, metadata


if __name__ == "__main__":
    transcribed_text, metadata = get_transcript_and_metadata(URL_OF_YOUTUBE_VIDEO)

    if transcribed_text and metadata:
        # Determine the directory of the current script
        current_script_directory = os.path.dirname(os.path.abspath(__file__))
        
        # Construct the output file path
        transcript_filename = os.path.join(current_script_directory, f"{metadata['video_id']}_transcript.txt")
        
        # Save the transcript as a text file in the same directory as the script
        with open(transcript_filename, 'w', encoding='utf-8') as file:
            file.write(transcribed_text)
        print(f"Transcript has been saved as {transcript_filename}")

    print("Original Text:")
    print(transcribed_text)

    print("\nMetadata:") 
    print(metadata)

    # New code to use timestamps:
    transcripts = metadata['transcripts']

    print("\nTimestamps:")
    for segment in transcripts:
        print(f"Text: {segment['text']}")
        print(f"Start: {segment['start']} seconds")
        print(f"Duration: {segment['duration']} seconds")
        print()

    # Filtering part as per asked timestamps
    start = 30  
    end = 60
    filtered = [seg for seg in transcripts if seg['start'] >= start and seg['start'] + seg['duration'] <= end]
    print(f"\nTranscript from {start} to {end} seconds:")
    for seg in filtered:
        print(seg['text'])

    # Now generate a summary of the full transcript using OpenAI
    print("\nGenerating summary...")
    summary, duration = get_summary_from_transcript(transcribed_text)
    print("Summary:")
    print(summary)
    print(f"Generated in {duration} seconds.")