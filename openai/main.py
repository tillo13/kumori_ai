from flask import Flask, render_template, request
from flask_socketio import SocketIO
from utils.socketio_utils import setup_event_handlers

# Define GCP project ID from environment variable
gcp_project_id = "kumori-404602"

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_very_secret_key'

# Initialize SocketIO with the Flask app
socketio = SocketIO(app, logger=True, engineio_logger=True)

# Setup event handlers for SocketIO
setup_event_handlers(socketio, gcp_project_id)

@app.route('/')
def welcome():
    # A welcome page with a "Begin" button
    return render_template("streaming/welcome.html")

@app.route('/stream_start')
def stream_start():
    # The existing page to enter text to start streaming
    return render_template("streaming/stream_start.html")

@app.route('/stream_end')
def stream_end():
    input_text = request.args.get('inputText', '') # Extract the query parameter
    return render_template("streaming/stream_end.html", version="1.0", page_title='Streaming', input_text=input_text)

if __name__ == "__main__":
    # Running the Flask app with Socket.IO support
    socketio.run(app, host="0.0.0.0", port=3000, debug=True)