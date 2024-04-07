# socketio_utils.py
from flask import request, copy_current_request_context, jsonify, json
from flask_socketio import SocketIO, emit, disconnect
from threading import Thread
from utils.openai_utils import OpenAIHelper
import logging

# Setup logger
logger = logging.getLogger('socketio')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Initialize the OpenAI helper with GCP project ID on_demand
def create_openai_helper(gcp_project_id):
    return OpenAIHelper(gcp_project_id)

# Helper function to set up SocketIO message handlers
def setup_event_handlers(socketio, gcp_project_id):


    @socketio.on('create_stream')
    def handle_create_stream(data):
        print("create_stream event received:", data)  # Debugging print statement

        openai_helper = create_openai_helper(gcp_project_id)
        session_id = request.sid
        messages = data.get('messages', [])
        # If 'gpt_override_model' is not provided by the frontend, default to 'gpt-4-1106-preview'
        gpt_override_model = data.get('gpt_override_model', 'gpt-4-1106-preview')

        logger.info('Creating stream with session id: %s', session_id)
        logger.info('Messages received for streaming: %s', messages)

        @copy_current_request_context
        def target():
            try:
                logger.info("Received messages for streaming: %s", messages)
                # Now pass the gpt_override_model, with the default being GPT-4 preview
                stream_openai_responses(socketio, openai_helper, session_id, messages, gcp_project_id, override_model=gpt_override_model)  
                emit('ack', {'success': True})
            except Exception as e:
                emit('ack', {'success': False, 'message': str(e)})

        Thread(target=target).start()

def setup_socketio(app):
    socketio = SocketIO(app, logger=True, engineio_logger=True)

    @socketio.on_error_default
    def error_handler(e):
        logger.error('SocketIO Error: %s', str(e))
        emit('error', {'message': 'An unexpected error has occurred.'})
    
    return socketio



def stream_openai_responses(socketio, openai_helper, session_id, messages, gcp_project_id, override_model=None):
    
    # Callback function to handle incoming messages from OpenAI
    def on_message(content):
        logger.info('Response from OpenAI: %s', content)
        # Emit a message event to the client
        socketio.emit('new_message', {'content': content}, room=session_id)

    # Callback function to handle errors during streaming
    def on_error(error_message):
        # Emit an error event to the client
        socketio.emit('stream_error', {'error': error_message}, room=session_id)

    # Callback function to close the connection cleanly
    def on_close():
        # Emit a stream complete event to the client
        socketio.emit('stream_complete', {'message': 'Stream completed.'}, room=session_id)
        disconnect(session_id)
    
    try:
        # Function doing the streaming operation
        openai_helper.stream_responses(messages, on_message, on_error, on_close, override_model)  

    except Exception as e:
        logger.error('Error while streaming OpenAI responses: %s', e)
        # Emit an error if the streaming operation fails
        emit('stream_error', {'error': str(e)})
    else:
        logger.info('Messages sent to OpenAI: %s', json.dumps(messages))

def on_stream_completion(socketio, session_id):
    try:
        logger.info(f'Emitting "stream_complete" for session_id: {session_id}')
        socketio.emit('stream_complete', {'message': 'Stream completed successfully'}, room=session_id)
    except Exception as e:
        logger.error(f'Streaming error: {e}')
        socketio.emit('stream_error', {'error': str(e)}, room=session_id)
