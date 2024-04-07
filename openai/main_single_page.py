from flask import Flask, render_template
from flask_socketio import SocketIO
from utils.openai_utils import OpenAIHelper
from utils.socketio_utils import setup_event_handlers
from os import environ as env
from dotenv import load_dotenv, find_dotenv

# Load environment variable
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

# Define GCP project ID from environment variable
gcp_project_id = "kumori-404602"

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_very_secret_key'

# Initialize SocketIO with the Flask app
socketio = SocketIO(app, logger=True, engineio_logger=True)

# Setup event handlers for SocketIO
setup_event_handlers(socketio, gcp_project_id)

@app.route('/stream')
def stream_page():
    return render_template("stream.html", version="1.0", page_title='Streaming')

if __name__ == "__main__":
    # Running the Flask app with Socket.IO support
    socketio.run(app, host="0.0.0.0", port=3000, debug=True)