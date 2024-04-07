import ollama
import os
import subprocess
import threading
import time
import platform
import psutil  
import signal
import time

# Capture the start time of the script
script_start_time = time.time()

SYSTEM_PROMPT = """
you're a helpful chatbot that helps with homework for a 6th grade student.  Greet the user and ask what homework they may need help with.
"""


# Hardcoded model choice
OLLAMA_MODEL = "wizard-vicuna-uncensored"

# Updated list of available voices including a subset of default voices and all the Siri voices
voices = [
    "Karen (Premium)", 
    "Zoe (Premium)",  
    "Matilda (Premium)",
    #"Joelle (Enhanced)",
    "Tessa (Enhanced)",
    "Siri", ]  

def get_available_voices():
    try:
        # Getting the list of available voices
        result = subprocess.run(['say', '-v', '?'], capture_output=True, text=True)
        available_voices = result.stdout
    except Exception as e:
        print(f"Failed to get available voices: {e}")
        available_voices = ""
    return available_voices

def choose_voice(voices):
    print("\nAvailable Voices:")
    for i, voice in enumerate(voices, 1):
        print(f"{i}. {voice}")
    choice = int(input("Choose a voice number: ")) - 1
    return voices[choice]

def validate_voice_choice(voice):
    available_voices = get_available_voices()
    if voice in available_voices:
        return True
    else:
        print(f"Voice `{voice}` not found. Falling back to the system's default voice.")
        return False

def speak_response(text, voice):
    """Utilizes the macOS 'say' command to voice the text with proper handling for special voices."""
    sanitized_text = text.replace('"', '\\"').replace("'", "\\'")  # Escape quotes in the text
    
    # Check for special 'Siri' naming without -v and without the extra quotes
    if voice == "Siri":
        # If selected voice is Siri, handle it differently as needed
        # Note: Adjust below as necessary for how you intend to handle Siri specifically
        # This is just a placeholder and might need adjustment based on how you handle Siri
        cmd = f'say "{sanitized_text}"'  # Potentially customize this for Siri
    else:
        # For all other cases, ensure the voice is quoted to handle spaces and parentheses properly
        cmd = f'say -v "{voice}" "{sanitized_text}"'
    
    os.system(cmd)

def speak_sentence_in_background(text, voice):
    """Speak given text sentence in the background."""
    thread = threading.Thread(target=speak_response, args=(text, voice))
    thread.start()

def is_complete_sentence(text):
    """Check if the text has a complete sentence."""
    # Simple heuristic: check for terminal punctuation marks
    return any(text.endswith(punct) for punct in ('.', '?', '!'))

def is_ollama_running():
    """Check if `ollama serve` is in the list of running processes."""
    try:
        result = subprocess.run(['pgrep', '-f', 'ollama serve'], capture_output=True, text=True)
        if result.stdout:
            return True
    except Exception as e:
        print(f"Error checking if Ollama is running: {e}")
    return False

# Global variable to store the PID
ollama_pid = None

def start_ollama():
    global ollama_pid
    try:
        # Start Ollama serve in a new Terminal window
        command = 'tell application "Terminal" to do script "ollama serve"'
        subprocess.run(['osascript', '-e', command])
        print("Starting Ollama service in a new Terminal window...")
        time.sleep(1)  # Give it a moment to start

        # Monitor for the newly started Ollama process
        time_started = time.time()
        while (time.time() - time_started) < 10:  # Adjust timeout as needed
            result = subprocess.run(['pgrep', '-f', 'ollama serve'], capture_output=True, text=True)
            if result.stdout:
                ollama_pid = int(result.stdout.strip().split('\n')[0])  # Takes the first PID if multiple found
                print(f"Ollama service started with PID: {ollama_pid}")
                break
            time.sleep(0.5)  # Check every half second
    except Exception as e:
        print(f"Failed to start Ollama service: {e}")

def end_ollama():
    try:
        # Find Ollama processes dynamically
        result = subprocess.run(['pgrep', '-f', 'ollama serve'], capture_output=True, text=True)
        if result.stdout:
            # Extract all PIDs
            ollama_pids = result.stdout.strip().split('\n')
            for pid in ollama_pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    print(f"Successfully terminated Ollama service with PID: {pid}")
                except Exception as e:
                    print(f"Failed to terminate Ollama service with PID: {pid} - {e}")
            global script_start_time
            session_duration = time.time() - script_start_time
            print(f"Session duration was {session_duration:.2f} seconds.")
        else:
            print("Ollama service is not running or no PIDs found.")
    except Exception as e:
        print(f"Failed to find Ollama processes: {e}")

def chat_with_ollama(chosen_voice):
    if not is_ollama_running():
        print("Ollama service not running. Attempting to start it...")
        start_ollama()

    print(f"Starting conversation with Ollama using the model: {OLLAMA_MODEL}.\n"
        "You can use the following commands:\n"
        "- Type 'stop' to stop the script.\n"
        "- Type '/end' to stop Ollama service completely and the conversation.\n"
        "- Type '/history' to view the conversation history.\n"
        "- Type '/swap' to swap the chat personality.\n")

    conversation_history = [{'role': 'system', 'content': SYSTEM_PROMPT}]

    # Initial chat call with Ollama
    speak_ollama_response(conversation_history, chosen_voice)
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "stop":
            print("Exiting conversation.")
            break
        elif user_input.lower() == "/end":
            end_ollama()  # Handle termination of Ollama
            print("Ending the chat session.")
            break
        elif user_input.lower() == "/history":
            # Print the conversation history
            print_conversation_history(conversation_history)
            continue
        elif user_input.lower() == "/swap":
            # Handle voice change
            print("\nSwapping chat personality...")
            chosen_voice = choose_voice(voices)  # Re-prompt and get the new voice
            print(f"Voice swapped to {chosen_voice}. Continuing conversation...\n")
            # No need to break or continue since we want to carry on the conversation
        else:
            if user_input:  # Ensure non-empty user input
                conversation_history.append({'role': 'user', 'content': user_input})
                speak_ollama_response(conversation_history, chosen_voice)
            else:
                print("Please enter a valid response.")

def print_conversation_history(conversation_history):
    print("\n--- Conversation History ---")
    for message in conversation_history:
        speaker = "System" if message['role'] == 'system' else "You" if message['role'] == 'user' else "Bot"
        print(f"{speaker}: {message['content']}")
    print("------------------------------\n")

def speak_ollama_response(conversation_history, chosen_voice):
    buffer = ""
    first_chunk = True  # Flag to identify the first chunk
    stream = ollama.chat(
        model=OLLAMA_MODEL,  # Model variable used here
        messages=conversation_history,
        stream=True,
    )
    
    for chunk in stream:
        message = chunk.get('message', {}).get('content', '')
        if first_chunk:
            # Append the chosen voice as a prefix to the message only for the first chunk
            print(f"{chosen_voice}: ", end='', flush=True)  # Print the voice name prefix only once
            first_chunk = False  # Reset the flag after the first chunk
        
        print(message, end='', flush=True)  # Print without the voice name prefix from the second chunk onwards
        
        buffer += message

        # When a complete sentence is formed
        if is_complete_sentence(buffer):
            speak_sentence_in_background(buffer, chosen_voice)
            # Append this part of the bot's response to the conversation history
            conversation_history.append({'role': 'assistant', 'content': buffer})
            buffer = ""  # Reset buffer after handling a complete sentence

    # After exiting the loop, handle any remaining text that didn't end with a proper punctuation mark
    if buffer.strip():
        # It could be a part of the bot's response not yet appended due to missing terminal punctuation.
        speak_sentence_in_background(buffer, chosen_voice)
        # Appending the final piece of the bot's message
        conversation_history.append({'role': 'assistant', 'content': buffer.strip()})
    
    # Assuming the last chunk contains the entire or final stats, collecting it here
    stats = {
        'done': chunk.get('done'),
        'total_duration': chunk.get('total_duration'),
        'load_duration': chunk.get('load_duration'),
        'prompt_eval_count': chunk.get('prompt_eval_count'),
        'prompt_eval_duration': chunk.get('prompt_eval_duration'),
        'eval_count': chunk.get('eval_count'),
        'eval_duration': chunk.get('eval_duration')
    }
    pretty_print_stats(stats, conversation_history)


def pretty_print_stats(stats, conversation_history):
    print("\n---- --------------------- ----")
    print("\n---- CONVERSATION METADATA ----")
    
    completed = 'Yes' if stats.get('done', False) else 'No'
    total_duration_sec = stats.get('total_duration', 0) / 1_000_000_000
    load_duration_sec = stats.get('load_duration', 0) / 1_000_000_000
    prompt_eval_count = stats.get('prompt_eval_count', 'N/A')
    prompt_eval_duration_sec = stats.get('prompt_eval_duration', 0) / 1_000_000_000
    eval_count = stats.get('eval_count', 'N/A')
    eval_duration_sec = stats.get('eval_duration', 0) / 1_000_000_000

    # Calculate engagement metrics based on conversation history
    num_user_messages = sum(1 for msg in conversation_history if msg['role'] == 'user')
    num_bot_messages = sum(1 for msg in conversation_history if msg['role'] == 'assistant')

    average_message_length_user = sum(len(msg['content']) for msg in conversation_history if msg['role'] == 'user') / max(1, num_user_messages)
    average_message_length_bot = sum(len(msg['content']) for msg in conversation_history if msg['role'] == 'assistant') / max(1, num_bot_messages)

    # Local Math Calculations
    average_response_time = (total_duration_sec - load_duration_sec) / max(1, num_bot_messages)
    conversation_turns = len(conversation_history) - 1  # Considering initial system prompt as not a turn

    # Formatting output
    average_response_time_str = f"{average_response_time:.3f} sec"
    average_message_length_user_str = f"{average_message_length_user:.2f} chars"
    average_message_length_bot_str = f"{average_message_length_bot:.2f} chars"
    conversation_turns = len(conversation_history) - 1  # Subtracting the initial system prompt

    # Calculate session duration
    session_duration_sec = time.time() - script_start_time    
    # Format session duration for display
    session_duration_str = f"{session_duration_sec:.3f} sec"

    # Combining all statistics
    stats_summary = (f"Session Duration: {session_duration_str} | " 
                    f"Completed: {completed} | "
                    f"Total Duration: {total_duration_sec:.3f} sec | "
                    f"Load Duration: {load_duration_sec:.3f} sec | "
                    f"Prompt Evaluation Count: {prompt_eval_count} | "
                    f"Prompt Evaluation Duration: {prompt_eval_duration_sec:.3f} sec | "
                    f"Evaluation Count: {eval_count} | "
                    f"Evaluation Duration: {eval_duration_sec:.3f} sec | "
                    f"Conversation Turns: {conversation_turns} | "  # Added here
                    f"Average Response Time: {average_response_time_str} | "
                    f"Average User Message Length: {average_message_length_user_str} | "
                    f"Average Bot Message Length: {average_message_length_bot_str}")
    
    print(stats_summary)
    print("--------------------------------\n")


def fetch_and_display_startup_info():
    python_version = platform.python_version()
    system_info = platform.uname()
    cpu_info = f"{psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical cores"
    mem_info = f"{psutil.virtual_memory().total / (1024**3):.2f} GB"

    print("Startup Information:")
    print(f"Python Version: {python_version}")
    print(f"System: {system_info.system}, Node: {system_info.node}, Version: {system_info.version}")
    print(f"CPU Info: {cpu_info}")
    print(f"Memory Info: {mem_info}")

    # Optionally, include Ollama Model details if running
    if is_ollama_running():
        print(f"Ollama Model Selected: {OLLAMA_MODEL}")
    else:
        print("Ollama service is not currently running.")

if __name__ == "__main__":
    # Instead of starting Ollama directly, check if it's running first
    if is_ollama_running():
        print("Ollama service is already running, we'll just use that instance.")
    else:
        os_platform = platform.system()
        if os_platform == "Darwin":  # Darwin is the system name for macOS
            start_ollama()
        else:
            print(f"Ollama auto-start is not supported on {os_platform}. Please manually start Ollama service.")
    
    # The rest of your startup logic remains the same
    fetch_and_display_startup_info()

    # Printing the SYSTEM_PROMPT to indicate the chosen bot's personality
    print("\nBot Personality: {}".format(SYSTEM_PROMPT))
    
    chosen_voice = choose_voice(voices)
    
    chat_with_ollama(chosen_voice)