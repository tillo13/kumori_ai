# openai_utils.py

import os
import json
import requests
from dotenv import load_dotenv

# Load OpenAI key from the .env file
load_dotenv()
openai_api_key = os.getenv("2023nov17_OPENAI_KEY")

# Define the endpoints
dalle_endpoint = "https://api.openai.com/v1/images/generations"
chat_endpoint = "https://api.openai.com/v1/chat/completions"

# ===IMAGE OPTIONS===
IMAGE_TO_CREATE = "A futuristic city skyline at sunset"
MODEL_VERSION = "dall-e-3"
NUMBER_OF_IMAGES = 1  # Must be between 1 and 10. For dall-e-3, only 1 is supported.
IMAGE_QUALITY = "standard"  # Can be 'standard' or 'hd' for dall-e-3
RESPONSE_FORMAT = "url"  # Must be 'url' or 'b64_json'
IMAGE_SIZE = "1024x1024"  # For dall-e-3, options are '1024x1024', '1792x1024', '1024x1792'
IMAGE_STYLE = "vivid"  # Can be 'vivid' or 'natural' for dall-e-3. 'vivid' is hyper-real, 'natural' is less so.
USER_ID = "unique_user_identifier"  # Optional, a unique identifier for your end-user.
# ===IMAGE OPTIONS===

# ===CHAT OPTIONS===
CHAT_MODEL = "gpt-3.5-turbo"
CHAT_TEMPERATURE = 0.5
CHAT_MAX_TOKENS = 450
CHAT_TOP_P = 1
CHAT_FREQUENCY_PENALTY = 0
CHAT_PRESENCE_PENALTY = 0
CHAT_STOP = None  # Up to 4 sequences where the API will stop generating further tokens.
CHAT_N = 1  # How many chat completion choices to generate for each input message.
CHAT_LOGPROBS = None  # Whether to return log probabilities of the output tokens or not.
CHAT_SEED = None  # If specified, our system will make a best effort to sample deterministically.
CHAT_USER_ID = USER_ID  # A unique identifier representing your end-user.
CHAT_STREAM = False  # If set, partial message deltas will be sent, like in ChatGPT.
CHAT_LOGIT_BIAS = None  # Modify the likelihood of specified tokens appearing in the completion.
CHAT_RESPONSE_FORMAT = None  # An object specifying the format that the model must output.
# ===CHAT OPTIONS===

# ===COST OPTIONS===
# OpenAI's pricing (as of the date provided in your example)
DALLE_PRICE_PER_IMAGE = 0.040  # Price per image for DALL·E 3 at standard quality and resolution
GPT_PRICE_PER_THOUSAND_TOKENS_INPUT = 0.0010  # Price per 1K tokens for GPT-3.5-turbo input
GPT_PRICE_PER_THOUSAND_TOKENS_OUTPUT = 0.0020  # Price per 1K tokens for GPT-3.5-turbo output
# ===COST OPTIONS===

def estimate_cost(number_of_images, number_of_input_tokens, number_of_output_tokens):
    """
    Estimate the cost of creation.

    :param number_of_images: Number of images that will be created by DALL-E.
    :param number_of_input_tokens: Total number of input tokens used in chat completions.
    :param number_of_output_tokens: Total number of output tokens generated in chat completions.
    :return: Estimated total cost.
    """
    image_cost = number_of_images * DALLE_PRICE_PER_IMAGE
    input_cost = (number_of_input_tokens / 1000) * GPT_PRICE_PER_THOUSAND_TOKENS_INPUT
    output_cost = (number_of_output_tokens / 1000) * GPT_PRICE_PER_THOUSAND_TOKENS_OUTPUT
    total_cost = image_cost + input_cost + output_cost
    return total_cost

def summarize_and_estimate_cost(summary_data):
    # Assuming summary_data contains the necessary information
    number_of_input_tokens = summary_data.get('total_input_tokens', 0)
    number_of_output_tokens = summary_data.get('total_output_tokens', 0)
    number_of_images = summary_data.get('number_of_images', 0)

    estimated_cost = estimate_cost(
        number_of_images=number_of_images,
        number_of_input_tokens=number_of_input_tokens,
        number_of_output_tokens=number_of_output_tokens
    )
    return estimated_cost

    summary = f"""
    ===SUMMARY===
    {summary_data.get('summary_text', '')}
    Estimated cost of creation: ${estimated_cost:.3f}
    """
    print(summary.strip())


def create_image(
        prompt,
        model="dall-e-3",
        n=1,
        quality="standard",
        response_format="url",
        size="1024x1024",
        style="vivid",
        user_id="unique_user_identifier"):
    
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "prompt": prompt,
        "model": model,
        "n": n,
        "quality": quality,
        "response_format": response_format,
        "size": size,
        "style": style,
        "user": user_id
    }

    print("Request payload:")
    print(json.dumps(payload, indent=4))

    try:
        response = requests.post(dalle_endpoint, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        print("Response payload:")
        print(json.dumps(response.json(), indent=4))

        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}


# Define the chat_completion function with all options

def chat_completion(
        messages,
        model=CHAT_MODEL,
        temperature=CHAT_TEMPERATURE,
        max_tokens=CHAT_MAX_TOKENS,
        top_p=CHAT_TOP_P,
        frequency_penalty=CHAT_FREQUENCY_PENALTY,
        presence_penalty=CHAT_PRESENCE_PENALTY,
        stop=CHAT_STOP,
        n=CHAT_N,
        logprobs=CHAT_LOGPROBS,
        seed=CHAT_SEED,
        user_id=CHAT_USER_ID,
        stream=CHAT_STREAM,
        logit_bias=CHAT_LOGIT_BIAS,
        response_format=CHAT_RESPONSE_FORMAT):

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stop": stop,
        "n": n,
        "logprobs": logprobs,
        "seed": seed,
        "stream": stream
    }

    if user_id is not None:
        payload["user"] = user_id

    if logit_bias is not None:
        payload["logit_bias"] = logit_bias

    if response_format is not None:
        payload["response_format"] = response_format

    print("Chat request payload:")
    print(json.dumps(payload, indent=4))

    try:
        response = requests.post(chat_endpoint, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        print("Chat response payload:")
        print(json.dumps(response.json(), indent=4))
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}

     


if __name__ == "__main__":
    # Example usage for chat_completion
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hello!"
        }
    ]
    
    # Configurable options for the chat API call
    chat_config_options = {
        "messages": conversation,
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 150,
        "top_p": 0.9,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.6,
        "stop": ["\n", "<|endoftext|>"],  # You can define more stopping criterias here if needed.
        "n": 1,
        "logprobs": 10,  # Or `None` if you don't want log probabilities.
        "seed": None,  # Or a specific integer if you want deterministic results.
        "user_id": "unique_user_identifier",
        "stream": False,  # Or `True` if you want partial outputs as the model generates them.
        "logit_bias": None,  # Or a dictionary with token biases if needed.
        "response_format": None  # Or {"type": "json_object"} for JSON responses.
    }

    chat_response = chat_completion(**chat_config_options)

    # Example usage for create_image, if desired, can be similar to this:
    image_config_options = {
        "prompt": IMAGE_TO_CREATE,
        "model": MODEL_VERSION,
        "n": NUMBER_OF_IMAGES,
        "quality": IMAGE_QUALITY,
        "response_format": RESPONSE_FORMAT,
        "size": IMAGE_SIZE,
        "style": IMAGE_STYLE,
        "user_id": USER_ID
    }
    
    image_response = create_image(**image_config_options)