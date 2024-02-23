import os
import time
import openai
from dotenv import load_dotenv

load_dotenv() 
OPENAI_API_KEY = os.getenv("2023nov17_OPENAI_KEY")
assert OPENAI_API_KEY, "Please make sure you have set `OPENAI_API_KEY` in your .env file."

# Initialise OpenAI secret key
openai.api_key = OPENAI_API_KEY

# Constants
MODEL_CHOICES = ["gpt-3.5-turbo", "text-davinci-003"]
#DEFAULT_MODEL = "text-davinci-003"  
DEFAULT_MODEL = "gpt-3.5-turbo"  
SYSTEM_MESSAGE = "You are acutely aware of recognizing names of people in videos, places they mention and things they describe.  Your job is to summarize youtube transcripts using bullet points to call out key data points"
MAX_TOKENS_RESPONSE = 500
MAX_TOKENS_SUMMARY = 500

def get_chat_response(prompt, model_choice=DEFAULT_MODEL):
    assert model_choice in MODEL_CHOICES, "Please make sure `model_choice` is in `MODEL_CHOICES`."
    start_time = time.time()
    response = get_response(prompt, model_choice)
    duration = time.time() - start_time
    return response, duration

def get_summary_from_transcript(transcript, model_choice=DEFAULT_MODEL):
    assert model_choice in MODEL_CHOICES, "Please make sure `model_choice` is in `MODEL_CHOICES`."
    start_time = time.time()
    summary = get_summary(transcript, model_choice)
    duration = time.time() - start_time
    return summary, duration

def get_response(prompt, model_choice):
    messages=[
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]

    if model_choice == MODEL_CHOICES[0]:
        response = openai.ChatCompletion.create(
            model=model_choice,
            messages=messages,
            temperature=0.2,
            max_tokens=MAX_TOKENS_RESPONSE
        )
        return response.choices[0].message['content'].strip()  # Moved this inside if for readability
    elif model_choice == MODEL_CHOICES[1]:
        prompt_string = f"{SYSTEM_MESSAGE}\n{prompt}"   # Davinci prefers single string prompts.
        response = openai.Completion.create(
            engine=model_choice,
            prompt=prompt_string,
            temperature=0.2,
            max_tokens=MAX_TOKENS_RESPONSE
        )
        return response.choices[0].text.strip()  # For Davinci, use text instead
    else:
        raise ValueError('Model not recognized. Please use one of the models in `MODEL_CHOICES`.')

def get_summary(transcript, model_choice):
    prompt = f"Here is a Youtube transcript for a video :\n\n{transcript}\n\nCould you provide a concise summary using bullet points?"
    messages=[
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]
  
    if model_choice == MODEL_CHOICES[0]:
        response = openai.ChatCompletion.create(
            model=model_choice,
            messages=messages,
            temperature=0.2,
            max_tokens=MAX_TOKENS_SUMMARY
        )
        summary = response.choices[0].message['content'].strip()  # Again, moved this inside if for readability
    elif model_choice == MODEL_CHOICES[1]:
        prompt_string = f"{SYSTEM_MESSAGE}\n{prompt}"   # Davinci prefers single string prompts.
        response = openai.Completion.create(
            engine=model_choice,
            prompt=prompt_string,temperature=0.2,
            max_tokens=MAX_TOKENS_SUMMARY
        )
        summary = response.choices[0].text.strip()  # For Davinci, use text
    else:
        raise ValueError('Model not recognized. Please use one of the models in `MODEL_CHOICES`.')

    return summary

if __name__ == "__main__":
    prompt = "What's a neat fact?"
    response, duration = get_chat_response(prompt)
    print("AI Response:")
    print(response)
    print(f"Generated in {duration} seconds.")
    print(f"OpenAI Model used was: {DEFAULT_MODEL}")