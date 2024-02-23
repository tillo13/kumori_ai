from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

# Initialize the model
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")

# The prompt you want to use
input_text = "Write me a poem about Machine Learning."

# Convert the prompt into tokens suitable for the model
input_ids = tokenizer(input_text, return_tensors="pt")

# Generate text using the model based on the tokens
outputs = model.generate(**input_ids)

# Decode the tokens back to a string
generated_text = tokenizer.decode(outputs[0])

# Print the generated text
print(generated_text)