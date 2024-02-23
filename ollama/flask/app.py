from flask import Flask, request, render_template, jsonify
import ollama
import subprocess
import time
import json


app = Flask(__name__)
script_start_time = time.time()

@app.route('/')
def index():
    # Rendering the HTML interface
    return render_template('ollama.html')

@app.route('/ollama')
def ollama():
    return render_template('ollama.html')

@app.route('/chat', methods=['POST'])
def chat():
    model_name = request.form['model']
    message = request.form['message']
    user_messages = [{'role': 'user', 'content': message}]

    # Interact with the chosen AI model
    response = interact_with_ollama(model_name, user_messages)
    return response


def interact_with_ollama(model_name, user_messages):
    response_start_time = time.time()

    # Check if the model is Gemma and handle differently
    if model_name.startswith("gemma"):
        response_text = interact_with_gemma(model_name, user_messages[-1]['content'])
    else:
        # Use the ollama library for models other than Gemma
        try:
            # We're assuming the user_messages list is how the ollama library expects it
            # You might need to adjust the data structure accordingly.
            stream_response = ollama.chat(
                model=model_name,
                messages=user_messages,
                stream=True
            )
            response_text = ""
            for chunk in stream_response:
                if 'message' in chunk and 'content' in chunk['message']:
                    response_text += chunk['message']['content']
        except Exception as e:  # Placeholder for catching specific exceptions
            response_text = f"An error occurred: {e}"

    response_end_time = time.time()

    response_json = {
        'model_response': response_text,
        'response_time': response_end_time - response_start_time,
        'total_time': response_end_time - script_start_time  # Make sure script_start_time is defined at the top of your file
    }
    return jsonify(response_json)

def interact_with_gemma(model_name, user_prompt):
    # Construct the command to invoke the Gemma model (make sure the command is correct)
    command = ['ollama', 'run', model_name]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout_data, stderr_data = process.communicate(input=user_prompt)
        process.wait()  # This should be redundant after communicate(), but it doesn't hurt to make sure.
        return stdout_data.strip()  # Remove any trailing newlines
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e.stderr}"

if __name__ == "__main__":
    app.run(debug=True)