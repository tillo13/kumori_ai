import openai
from .google_secret_utils import get_secret_version
from openai.error import OpenAIError
import json
import logging
import requests
import traceback

# Make sure to handle OpenAIError properly by importing it
from openai.error import OpenAIError
from .postgres_utils import get_db_connection, log_openai_usage, fetch_model_cost




class OpenAIHelper:
    def __init__(self, gcp_project_id):
        self.gcp_project_id = gcp_project_id
        # Store both model identifiers
        self.fallback_model = "gpt-4-1106-preview"
        self.primary_model = "gpt-3.5-turbo"
        self.primary_model = "gpt-4-1106-preview"
        # Initially, use the primary model
        self.model = self.primary_model
        # uncomment this to swap if needed self.model = self.fallback_model
        #get_secret_version function properly retrieves the API key
        openai.api_key = get_secret_version(self.gcp_project_id, 'KUMORI_OPENAI_KEY')
        openai.organization = get_secret_version(self.gcp_project_id, 'KUMORI_OPENAI_ORG_ID')

        # Class variables for the vision model
        self.AI_VISION_MODEL = "gpt-4-vision-preview"
        self.AI_VISION_IMAGE_DESCRIPTION_PROMPT = "If there are any people in this, it's crucial to describe all the colors of their appearance including clothes, hair and unique features. If no humans, describe in detail still so an ai generator can best replicate."

    def create_summary(self, text, max_tokens=300, model_type=None, celebrity_style=None):
        # system_message = f"You are a powerful summarizer capable of imitating an individual people when given a name."
        # user_message = f"Please summarize the following identically like {celebrity_style} would: {text}"
        system_message = f"You are an AI capable of exactly mimicking the writing style and tone of a persona in 2 or 3 sentences max."
        user_message = f"Assuming the role of {celebrity_style}, extract compelling points from the following text to construct a LinkedIn post in their exact voice that showcases thought leadership in the field, invites discussion, and includes relevant hashtags: {text}"
    
        # If a model_type parameter has been provided, use that.  
        # Otherwise, use self.primary_model as the default.
        model_to_use = model_type if model_type else self.primary_model

        try:
            response = openai.ChatCompletion.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=max_tokens, 
                temperature=0.7, 
            )
            logging.debug(f"OpenAI API response: {response}")

            summary = response.choices[0].message['content'].strip()
            model_used = model_to_use
            return summary, model_used

        except OpenAIError as e:
            logging.error(f"OpenAI API error: {e}")
            try:
                model_to_use = self.fallback_model
                response = openai.ChatCompletion.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.7,   
                )
                summary = response.choices[0].message['content'].strip()
                model_used = model_to_use
                return summary, model_used

            except OpenAIError as e:
                logging.error(f"OpenAI API error with fallback model: {e}")
                raise


    def moderate_content(self, content):
        try:
            response = openai.Moderation.create(input=content)
            return response, self.model
        except OpenAIError as e:
            logging.error(f"OpenAI API error occurred when calling Moderate Content: {str(e)}")
            raise
        except Exception as ex:
            logging.error(f"Unexpected exception occurred in moderate_content: {str(ex)}")
            raise


    #first we fetch current costs: 
    def calculate_openai_cost(self, model, tokens_used, cost_type='output'):
        cost_per_1k_tokens = fetch_model_cost(self.gcp_project_id, model, cost_type)
        if cost_per_1k_tokens is None:
            # If no cost is found, use a default value and log this event for investigation.
            cost_per_1k_tokens = 0.00
            logging.warning(f"Default cost used for unknown model '{model}' and cost type '{cost_type}'. Investigation needed.")
            
        cost = (cost_per_1k_tokens / 1000) * tokens_used
        return cost, self.model
   


    def describe_image_with_gpt_vision(self, image_data, image_type="url", max_tokens=1000, auth_provider_id=None, model=None, custom_prompt=None):


            
    
            #set model if passed from main.py else keep default
            model = model or self.AI_VISION_MODEL

            # Choose the custom prompt or fall back to the default AI vision description prompt
            prompt = custom_prompt if custom_prompt else self.AI_VISION_IMAGE_DESCRIPTION_PROMPT


            # Verify that image_type is either 'url' or 'base64'
            if image_type not in ['url', 'base64']:
                raise ValueError("image_type must be either 'url' or 'base64'")
            
            try:
                import time
                start_time = time.time()

                # Set up image content based on the type
                image_content = ({"type": "image_url", "image_url": image_data}
                                if image_type == "url"
                                else {"type": "image_url", "image_url": "data:image/jpeg;base64," + image_data})

                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                image_content
                            ]
                        }
                    ],
                    max_tokens=max_tokens
                )

                response_time = time.time() - start_time
                logging.info(f"Received response from OpenAI API: {response}")

                tokens_used = response.get('usage', {}).get('total_tokens', max_tokens)

                model_cost = fetch_model_cost(self.gcp_project_id, 'gpt-4-vision-preview', 'input')
                if model_cost is None:
                    logging.warning("No pricing found for model 'gpt-4-vision-preview' with cost type 'input'. Using default cost of $0.00.")
                    model_cost = 0.00  # Use a default cost of $0.00 when there is no pricing

                cost = model_cost / 1000 * tokens_used

                # Extract the description by looking for the first message with 'text' type content.
                if 'choices' in response and response['choices']:
                    for choice in response['choices']:
                        if 'message' in choice:
                            message = choice['message']
                            logging.info(f"Parsed message: {message}")

                            if isinstance(message['content'], list):
                                for content in message['content']:
                                    if 'text' in content:
                                        description = content['text']
                                        # Include the model in the return statement
                                        return description.strip(), response_time, tokens_used, cost, self.model
                            elif isinstance(message['content'], str):
                                # Include the model in the return statement
                                return message['content'].strip(), response_time, tokens_used, cost, self.model

                raise ValueError("Valid image description not found in the response")

            except openai.error.OpenAIError as e:
                logging.error(f"OpenAI API error occurred. Request ID: {e.request_id}. Details: {e}")
                raise e
            except Exception as e:
                logging.error(f"Unexpected error occurred in describe_image_with_gpt_vision: {traceback.format_exc()}")
                raise e

        





    def stream_responses(self, messages, on_message, on_error, on_close=None, override_model=None):
        # Use the provided override model, if given, otherwise fall back to the default model
        model_to_use = override_model if override_model else self.model

        try:
            # Log the messages being sent to OpenAI
            logging.info("Starting streaming to OpenAI with messages: %s", messages)
            
            # Your existing streaming logic
            stream = openai.ChatCompletion.create(
                model=model_to_use,
                messages=messages,
                max_tokens=2048,
                stream=True,
            )

            logging.info("OpenAI stream created successfully.")

            # Process messages from the stream
            for message in stream:
                logging.info("Received message from OpenAI: %s", message)
                if "choices" in message and "delta" in message["choices"][0] and \
                        "content" in message["choices"][0]["delta"]:
                    content = message["choices"][0]["delta"]["content"]
                    on_message(content)

            # Call the on_close callback if it is provided
            if on_close:
                logging.info("Streaming is completed. Calling on_close callback.")
                on_close()


        except OpenAIError as e:
            logging.error("An error occurred with the OpenAI streaming API: %s", str(e))
            on_error(str(e))
        except Exception as e:
            # This captures any other errors
            logging.error(
                "An unexpected error occurred in OpenAIHelper.stream_responses: %s",
                str(e)
            )
            on_error("An unexpected error occurred.")
            # You may also want to trigger on_close() here if relevant
            if on_close:
                logging.error("Triggering on_close callback due to an unexpected error.")
                on_close()
        return self.model


    def create_personalized_ai_text(self, user_messages, override_gpt_model=None):
        # Use the provided model if specified, otherwise fall back to the default model
        model_to_use = override_gpt_model if override_gpt_model else self.model

    
        try:
            response = openai.ChatCompletion.create(
                model=model_to_use,
                messages=user_messages,
                max_tokens=1000,
                temperature=0.7
            )
            personalized_text = response.choices[0].message['content'].strip()
            return personalized_text, model_to_use
        except OpenAIError as e:
            logging.error(f"OpenAI API error: {e}")
            raise e








#this is for the get_started path specifically
    def try_it_now(self, writing_style, latest_news_article=None):
        # Here you should craft the system message to include the writing style in a meaningful way.
        system_message = "It is critical to prioritize and closely emulate the writer's unique style as outlined in the writing style analysis. With this focus, use the details from the recent news article to craft an engaging 3-5 sentence LinkedIn post designed to drive engagement."


        # Set up default article details in case no article is fetched
        default_article = {
            'title': 'noArticleMatched',
            'source': 'noArticleMatched',
            'category': 'noArticleMatched',
            'description': 'No recent articles were found to match the news preferences. Please create a generic post about technology.',
            'url': 'noArticleMatched'
        }
        
        # Use the fetched article if available, otherwise use the default values
        article_details = latest_news_article if latest_news_article else default_article

        # Construct the messages including the writing style and article details
        user_messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Writing style analysis: {writing_style}"},
            {"role": "user", "content": f"Article title: {article_details['title']}"},
            {"role": "user", "content": f"Article source website: {article_details['source']}"},
            {"role": "user", "content": f"Article genre category: {article_details['category']}"},
            {"role": "user", "content": f"Article description: {article_details['description']}"},
            {"role": "user", "content": f"Article URL: {article_details['url']}"}
        ]

        try:
            # Use ChatCompletion endpoint with the revised prompt
            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=user_messages,
                max_tokens=1000,
                temperature=0.7,
                user="linkedin-post-user"
            )

            # Assuming the last message from the assistant should be the LinkedIn post
            linkedin_post = completion.choices[0].message['content'].strip()

            # At this point, you successfully received the LinkedIn post content
            return linkedin_post, self.model

        except OpenAIError as e:
            # Log both the simple error message and the full traceback for diagnostics
            logging.error(f"OpenAI API error occurred when calling try_it_now: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            # You may choose to handle the exception differently here,
            # or re-raise to allow a higher-level handler to deal with it.
            raise

        except Exception as ex:
            # Capture any other exception that could occur, logging the message and traceback
            logging.error(f"Unexpected exception occurred in try_it_now: {str(ex)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            # Re-raise the exception after logging
            raise

    
    def create_story(self, get_started_entry):
        # Construct the prompt for the chat as a conversation with the model
        system_message = "You are a well-informed assistant. Write a creative story."
        user_message = f"Create a story for the {get_started_entry['industry']} industry, \
                        which aims to {get_started_entry['content_objective']} through a \
                        {get_started_entry['content_platform']} platform. The tone should be \
                        {get_started_entry['brand_tone']}, focusing on the topic \
                        '{get_started_entry['keywords']}'. Incorporate the following writing style: \
                        {get_started_entry['writing_style']}."

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}],
                max_tokens=4096,
                temperature=0.7
            )
            # Assuming response.choices[0] is the generated story
            return response.choices[0].message['content'].strip() if response.choices else None
        except OpenAIError as e:
            logging.error(f"An error occurred using OpenAI to create a story: {e}")
            raise e

    # ...assess a users writing style
    def assess_writing_style(self, writing_sample):
        # System message to set the assistant's behavior for analyzing text
        system_message = (
            "Analyze the provided writing sample and describe the author's writing style in terms of "
            "vocabulary, sentence structure, grammar, punctuation, tone and mood, point of view, "
            "dialogue style, level of detail in descriptions, and themes. Provide concise, actionable "
            "insights that can be used to mimic the style."
        )
        
        # User message that contains the writing sample
        user_message = {"role": "user", "content": writing_sample}
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model, 
                messages=[
                    {"role": "system", "content": system_message},
                    user_message
                ]
            )
            
            # Since it's chat, you may receive a list of messages, where the last one should be the response
            if response and response.get('choices'):
                # Using the index -1 to get the last message in the chat response list
                result = response['choices'][0]['message']['content'].strip()
                return result, self.model
            else:
                raise ValueError("Unexpected response format.")
        except OpenAIError as e:
            # Implement your error handling (e.g., logging)
            raise e




    def ask(self, messages, model_type=None, temperature=0.7, max_tokens=1000):
        model_to_use = model_type if model_type else self.primary_model
        current_tokens = sum([len(message['content']) for message in messages]) // 4
        max_allowed_tokens = 4096
        max_tokens = min(max_tokens, max_allowed_tokens - current_tokens)
        
        try:
            response = openai.ChatCompletion.create(
                model=model_to_use,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response['choices'][0]['message']['content'].strip()
            tokens_used = response['usage']['total_tokens']  # Extracting tokens used
            return {'message': content}, tokens_used, model_to_use
        except OpenAIError as e:
            logging.error(f"Error with model {model_to_use}: {e}")
            raise e


        

    def format_response(self, message, model_version):
        # Check if the response from the model is already JSON-formatted
        if isinstance(message, dict):
            content = message.get('content')
            if isinstance(content, str) and content.startswith('{') and content.endswith('}'):
                # Try to load the content as JSON; if it fails, treat it as a simple string
                try:
                    content = json.loads(content)
                    # Extract 'response' key from JSON content if available
                    if 'response' in content:
                        message = f"{model_version} says: {content['response']}"
                    elif 'content' in content:
                        message = f"{model_version} says: {content['content']}"
                    else:
                        message = f"{model_version} says: {json.dumps(content)}"
                except json.JSONDecodeError:
                    # The content is not valid JSON, treat it as a simple string
                    message = f"{model_version} says: {content}"
            else:
                # The content key exists and is a string not in JSON format, prepend model version
                message = f"{model_version} says: {content}"
        else:
            # The model's message is not a dictionary (or not expected structure)
            raise ValueError("Response format not recognized")

        return {'content': message}, self.model


    @staticmethod
    def parse_openai_response(response):
        # Checks if the response has choices and a message/delta
        if 'choices' in response and len(response['choices']) > 0:
            choice = response['choices'][0]

            # Check if it's a regular message format
            if 'message' in choice and 'content' in choice['message']:
                return choice['message']['content']

            # Check if it's a 'delta' chunk format
            elif 'delta' in choice and 'content' in choice['delta']:
                return choice['delta']['content']

        raise ValueError("Response format not recognized")
    
    ######DALLE2 and 3 specific endpoints as of 2023nov17############
    def create_image(self, prompt, model='dall-e-2', n=1, size='1024x1024', response_format='url', user=None, style=None, quality=None):
        """
        Creates an image given a prompt, with support for both DALL-E 2 and DALL-E 3.
        :param model: The model to use for image generation ('dall-e-2' or 'dall-e-3')
        :param prompt: Text description of the desired image(s)
        :param n: The number of images to generate
        :param size: The size of the generated images
        :param quality: The quality of the image that will be generated
        :param style: The style of the generated images
        :param response_format: The format in which the generated images are returned
        :param user: A unique identifier representing your end-user
        :return: The JSON response containing the image url or base64 content
        """
        endpoint = "https://api.openai.com/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {openai.api_key}",
            "OpenAI-Organization": openai.organization
        }

        # Print information about the model and character count
        prompt_length = len(prompt)
        print(f"Using the model: {model}")
        print(f"Character count of the prompt: {prompt_length}")

        # Prepare the data payload with optional parameters
        data = {
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": response_format
        }

        # Add model-specific parameters
        if model == 'dall-e-3':
            data.update({
                "model": model,
                "style": style,
                "quality": quality
            })

        # Include the user identifier in the data payload if provided
        if user:
            data['user'] = user

        # Print the payload for debugging
        print("Sending payload:")
        print(json.dumps(data, indent=2))
        
        try:
            response = requests.post(endpoint, headers=headers, json=data)
            response.raise_for_status()
            response_json = response.json()
            print("Received response:")
            print(json.dumps(response_json, indent=2))
            return response_json, self.model
        # Adjusted error handling
        except requests.HTTPError as http_err:
            # Response text may not exist or may be empty, so log it regardless
            error_body = getattr(http_err.response, 'text', 'No response body available')
            logging.error(f"HTTP error occurred: {http_err}, response body: {error_body}")
            
            # If 'text' is not available or is empty, try accessing 'content'
            if not error_body or error_body == 'No response body available':
                try:
                    # Attempt to decode the response content as JSON to provide structured error info
                    error_detail = http_err.response.json()
                    logging.error(f"Full JSON response: {error_detail}")
                except ValueError:
                    # Fall back to raw bytes if the response is not JSON formatted
                    error_detail = http_err.response.content
                    logging.error(f"Raw response content: {error_detail}")

            # Re-raise the exception with original HTTPError content
            raise http_err
        except Exception as err:
            # For other exceptions, include as much detail in the logs as possible.
            logging.error(f"An error occurred: {err}")
            # You might want to log the traceback here as well
            logging.error(traceback.format_exc())  # This prints the full traceback of the error
            raise err
        


















        #####untested below here as of 2023dec6#######

    def create_image_edit(self, image_path, prompt, mask_path=None, n=1, size='1024x1024', response_format='url', user=None):
            """
            Creates an edited or extended image given an original image and a prompt.

            :param image_path: Path to the image file to edit
            :param prompt: Text description of the desired modifications
            :param mask_path: Optional path to the mask image file
            :param n: The number of images to generate
            :param size: The size of the generated images
            :param response_format: The format in which the generated images are returned
            :param user: A unique identifier representing your end-user
            :return: The JSON response containing the edited image url or base64 content
            """
            endpoint = "https://api.openai.com/v1/images/edits"
            headers = {
                "Authorization": f"Bearer {openai.api_key}",
                 "OpenAI-Organization": openai.organization
            }
            files = {
                "prompt": (None, prompt),
                "n": (None, str(n)),
                "size": (None, size),
                "response_format": (None, response_format)
            }
            
            with open(image_path, 'rb') as image_file:
                files['image'] = ('image.png', image_file)  # The 'image.png' is a placeholder filename

                if mask_path:
                    with open(mask_path, 'rb') as mask_file:
                        files['mask'] = ('mask.png', mask_file)  # The 'mask.png' is a placeholder filename

                # Include the user identifier in the request if provided
                if user:
                    files['user'] = (None, user)

                try:
                    response = requests.post(endpoint, headers=headers, files=files)
                    response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
                    return response.json(), self.model
                except requests.HTTPError as http_err:
                    # Handle HTTP errors
                    logging.error(f"HTTP error occurred: {http_err}")
                    raise http_err
                except Exception as err:
                    # Handle other errors
                    logging.error(f"An error occurred: {err}")
                    raise err

    def create_image_variation(self, image_path, n=1, size='1024x1024', response_format='url', user=None):
            """
            Creates a variation of a given image.

            :param image_path: Path to the image file to use for variation
            :param n: The number of variations to generate
            :param size: The size of the generated variations
            :param response_format: The format in which the generated variations are returned
            :param user: A unique identifier representing your end-user
            :return: The JSON response containing the variation image url or base64 content
            """
            endpoint = "https://api.openai.com/v1/images/variations"
            headers = {
                "Authorization": f"Bearer {openai.api_key}",
                 "OpenAI-Organization": openai.organization
            }
            files = {
                "n": (None, str(n)),
                "size": (None, size),
                "response_format": (None, response_format)
            }

            # Open the image file using a context manager
            with open(image_path, 'rb') as image_file:
                files['image'] = ('image.png', image_file)  # The 'image.png' is a placeholder filename

                # Include the user identifier in the request if provided
                if user:
                    files['user'] = (None, user)

                try:
                    response = requests.post(endpoint, headers=headers, files=files)
                    response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
                    return response.json(), self.model
                except requests.HTTPError as http_err:
                    # Handle HTTP errors
                    logging.error(f"HTTP error occurred: {http_err}")
                    raise http_err
                except Exception as err:
                    # Handle other errors
                    logging.error(f"An error occurred: {err}")
                    raise err