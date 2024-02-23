# interact_with_model.py
from chat_engine import get_response_from_model

# Set your model and message internally
model_name = 'mistral'  # The model you want to use
user_message = 'How are you today?'  # The message you want to send

# Now, you can call the get_response function from chat.py with these values
response = get_response_from_model(model_name, user_message)
print(f"The model {model_name} responds: {response}")