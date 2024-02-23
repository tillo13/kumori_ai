"""
2024feb23 # Notes:
This script allows you to chat with different AI models using the Ollama client.
Before you run the script for the first time, make sure to pull the AI models using the following commands:

ollama pull llama2      # To pull the Llama2 model
ollama pull gemma:2b    # To pull the Gemma 2B model
ollama pull gemma:7b    # To pull the Gemma 7B model
ollama pull mistral     # To pull the Mistral model

If you encounter the 'model not found' error, it's likely that the model needs to be pulled first.

# Usage:
Run the script in a terminal with the command `python chat.py` and follow the interactive prompt to chat with different models.
"""

import ollama
import subprocess
import time
import threading

# Track script start time
script_start_time = time.time()

# Define the available models
MODELS = {
    '1': 'llama2',
    '2': 'gemma:2b',
    '3': 'gemma:7b',
    '4': 'mistral',
}

def choose_model():
    print("Please choose a model to talk with:")
    for key, value in MODELS.items():
        print(f"{key}. {value}")
    
    choice = input("Enter the number of the model: ")
    return MODELS.get(choice, 'llama2')  # Default to 'llama2' if an invalid choice is made

def timed_print(start_time, interval=0.01):
    """Print the elapsed time since the start every interval seconds."""
    elapsed = time.time() - start_time
    print(f"\rWaiting for response: {elapsed:.2f}s", end="", flush=True)
    global timer_thread
    timer_thread = threading.Timer(interval, timed_print, [start_time])
    timer_thread.start() 

def create_gemma_command(model_name):
    """Create the command to run the Gemma model."""
    command = ['ollama', 'run', model_name]
    return command

def print_formatted(message, payload=None, overwrite_line=False):
    if overwrite_line:
        print("\r", " " * 60, "\r", end='')  # Clear the line if needed
    if payload is not None:
        print(f"===\n{message}\n{payload}\n===")
    else:
        print(message)

# Use `with` statement for subprocess to ensure proper resource cleanup
def start_gemma_subprocess(command, user_prompt):
    with subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    ) as process:
        return process.communicate(input=user_prompt)

# Updated `handle_gemma_response` to use the new `print_formatted` function
def handle_gemma_response(start_time, stdout):
    global timer_thread
    timer_thread.cancel()  # Stop the timed print
    end_time = time.time()
    response_time_message = f"Response in {end_time - start_time:.2f} seconds"
    response_payload = {"response": stdout}
    print_formatted(response_time_message, response_payload, overwrite_line=True)
    return stdout

# Updated `handle_gemma_subprocess_error` to use the new `print_formatted` function
def handle_gemma_subprocess_error(e):
    global timer_thread
    timer_thread.cancel()  # Stop the timed print
    print_formatted(f"An error occurred: {e.stderr}")
    return f"An error occurred: {e.stderr}"

def interact_with_gemma(model_name, user_prompt):
    global timer_thread
    start_time = time.time()
    timed_print(start_time)  # Start the timed print
    
    command = create_gemma_command(model_name)
    payload = {"command": command, "User prompt": user_prompt}
    print_formatted("Request JSON payload (Sent to Gemma):", payload)
    
    try:
        stdout, stderr = start_gemma_subprocess(command, user_prompt)
        return handle_gemma_response(start_time, stdout)
    except subprocess.CalledProcessError as e:
        return handle_gemma_subprocess_error(e)

def create_request_json(model_name, user_messages):
    """Create the JSON payload for the request."""
    request_json = {
        "model": model_name,
        "messages": user_messages
    }
    return request_json


def process_ollama_response(model_name, user_messages):
    """Process the response from the Ollama model."""
    try:
        # Use the ollama library for models other than Gemma
        return ollama.chat(model=model_name, messages=[user_messages[-1]], stream=True)
    except ollama._types.ResponseError as e:
        print(f"Error: {e.args[0]}")
        print("It seems the model has not been pulled yet. Please pull the model using the following command:")
        print(f"ollama pull {model_name}")
        return None  # Return None to indicate that an error occurred

def print_ollama_response(model_name, stream_response):
    """Print the response from the Ollama model."""
    first_chunk_received = False
    for chunk in stream_response:
        if 'message' in chunk and 'content' in chunk['message']:
            if not first_chunk_received:
                print_response_payload(model_name, chunk)
                print("Bot: ", end='', flush=True)
                first_chunk_received = True
            print(chunk['message']['content'], end='', flush=True)
    print("\n")  # Ensure newline after response

def print_response_payload(model_name, chunk):
    """Print the JSON payload for the response."""
    response_json = {
        "model": model_name,
        "chunk": chunk
    }
    print("===")
    print(f"Response JSON payload (Received from Ollama):\n{response_json}")
    print("===")

def interact_with_ollama(model_name, user_messages):
    response_start_time = time.time()  # Start time for the response
    request_json = create_request_json(model_name, user_messages)
    print_formatted("Request JSON payload (Sent to Ollama):", request_json)
    
    if model_name.startswith("gemma"):
        # Call interact_with_gemma instead of the nonexistent process_gemma_response
        model_response = interact_with_gemma(model_name, user_messages[-1]['content'])
        print(f"Gemma: {model_response}")
    else:
        stream_response = process_ollama_response(model_name, user_messages)
        if stream_response is not None:
            print_ollama_response(model_name, stream_response)
        else:
            return None  # Return None if there was an error

    response_end_time = time.time()  # End time for the response
    response_time = response_end_time - response_start_time
    total_time = response_end_time - script_start_time
    # Print breadcrumb footer
    print(f"-----\nBreadcrumb-> Model used: {model_name} | Response time: {response_time:.2f} seconds | "
          f"Total time since beginning script: {total_time:.2f} seconds\n-----\n")
    return response_time  # For breadcrumb timing in the main conversation loop

def chat_with_model(model):
    print(f"Let's chat with the model '{model}'. Type 'quit' to exit or 'switch' to change the model.")
    conversation_history = []
    while True:
        user_input = input("You: ")
        user_input_lower = user_input.lower()
        if user_input_lower == 'quit':
            break
        elif user_input_lower == 'switch':
            # Reset conversation history and prompt for model selection
            conversation_history = []
            print("\nSwitching models...")
            return choose_model()
        conversation_history.append({'role': 'user', 'content': user_input})
        # Capture response time for breadcrumb footer
        response_time = interact_with_ollama(model, conversation_history)

if __name__ == "__main__":
    model_to_chat = choose_model()
    while model_to_chat:
        model_to_chat = chat_with_model(model_to_chat)