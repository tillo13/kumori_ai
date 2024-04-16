import os
import subprocess
import threading
import sys

def run_animation_in_new_terminal():
    """Run the bot animation Python script in a new Terminal window."""
    animation_script_path = os.path.abspath("bot_animation.py")  # Ensure this is the correct path to your animation script
    command = f"""osascript -e 'tell application "Terminal" to do script "python3 \"{animation_script_path}\""'"""
    subprocess.run(command, shell=True)


# List of available voices for the chatbot
voices = [
    "Karen (Premium)", 
    "Zoe (Premium)",  
    "Matilda (Premium)",
    #"Joelle (Enhanced)",
    "Tessa (Enhanced)",
    ]  

CHAT_SPEED = 200  # Adjust speaking rate as needed

def run_animation_in_new_terminal():
    """Run the bot animation Python script in a new Terminal window."""
    animation_script_path = os.path.abspath("bot_animation.py")  # Ensure this path is to your animation script
    command = f"""osascript -e 'tell application "Terminal" to do script "python3 \\"{animation_script_path}\\""'"""
    subprocess.run(command, shell=True)

def choose_voice():
    """Prompt the user to choose a voice from the list."""
    print("\nAvailable Voices:")
    for i, voice in enumerate(voices, 1):
        print(f"{i}. {voice}")
    choice = 0  # Default choice
    while True:
        try:
            choice = int(input("Choose a voice number: ")) - 1
            if 0 <= choice < len(voices):
                break
            else:
                print("Invalid choice. Please select a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    return voices[choice]

def speak(text, voice):
    """Speak out the text using the macOS 'say' command with the chosen voice."""
    sanitized_text = text.replace('"', '\\"').replace("'", "\\'")  # Escape quotes in the text
    cmd = f'say --rate {CHAT_SPEED} -v "{voice}" "{sanitized_text}"'
    os.system(cmd)

def main():
    run_animation_in_new_terminal()  # Launch animation in its own window
    print("Welcome! Let's test the ASCII animation chatbot.")
    chosen_voice = choose_voice()
    speak("Hello there! How are you today?", chosen_voice)

if __name__ == "__main__":
    main()