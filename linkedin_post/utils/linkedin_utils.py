import requests
import os
from flask import redirect, url_for
from werkzeug.utils import secure_filename

from .google_secret_utils import get_secret_version

# Directly set the ACTIVE_ENV here development or production
ACTIVE_ENV = 'development'  

# Define the REDIRECT_URIs directly
LOCAL_REDIRECT_URI = 'http://127.0.0.1:5000/linkedin/callback'
PROD_REDIRECT_URI = 'https://kumori.ai/linkedin/callback'

# Choose the appropriate REDIRECT_URI based on the ACTIVE_ENV
REDIRECT_URI = PROD_REDIRECT_URI if ACTIVE_ENV == 'production' else LOCAL_REDIRECT_URI

# Set the Google Cloud Project ID
PROJECT_ID = 'kumori-404602'

# Retrieve the secrets from Google Secret Manager
CLIENT_ID = get_secret_version(PROJECT_ID, 'KUMORI_LINKEDIN_CLIENT_ID')
CLIENT_SECRET = get_secret_version(PROJECT_ID, 'KUMORI_LINKEDIN_PRIMARY_CLIENT_SECRET')

def redirect_to_linkedin_auth_url():
    return redirect(generate_linkedin_auth_url())

def process_linkedin_callback(args):
    code, error = args.get('code'), args.get('error')
    if error or not code:
        return {'error': f'Error receiving authorization code: {error}', 'status_code': 400}
    
    access_token, error_msg, status_code = get_linkedin_access_token(code)
    if access_token:
        user_info, sub_value = fetch_linkedin_user_info(access_token)
        data = {'sub': sub_value, 'access_token': access_token, 'json_response': user_info}
        return {'data': data}
    else:
        return {'error': error_msg, 'status_code': status_code}

def post_to_linkedin_with_image(request):
    form = request.form
    file = request.files.get('image_file')
    result = handle_linkedin_image_and_post(form, file)
    
    # Consider that handle_image_and_post might return a dict with either 'error' or 'message'
    return result

def handle_linkedin_image_and_post(form, file):
    text_to_post = form['post_text']
    sub_value = form['sub']
    access_token = form['access_token']
    image_urn = None

    # Handle image upload if file is present
    if file and file.filename:
        filename = secure_filename(file.filename)
        file_path = os.path.join('/tmp', filename)
        file.save(file_path)

        image_urn_resp = upload_image_to_linkedin(file_path, sub_value, access_token)
        if isinstance(image_urn_resp, tuple):
            return {'error': image_urn_resp[0], 'status_code': image_urn_resp[1]}

        image_urn = image_urn_resp

    post_response = post_to_linkedin(text_to_post, sub_value, access_token, image_urn)
    if isinstance(post_response, tuple):
        return {'error': post_response[0], 'status_code': post_response[1]}

    message = f'Posted to LinkedIn with text: {text_to_post}'
    if image_urn:
        message += f' and image: {image_urn}'
    return {'message': message}

def generate_linkedin_auth_url():
    auth_url = 'https://www.linkedin.com/oauth/v2/authorization'
    params = {
        'response_type': 'code',
        'client_id': CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'scope': 'openid profile email w_member_social',
        'state': 'randomstring'
    }
    url = requests.Request('GET', auth_url, params=params).prepare().url
    return url

def get_linkedin_access_token(code):
    token_url = 'https://www.linkedin.com/oauth/v2/accessToken'
    data = {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': REDIRECT_URI,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET
    }
    response = requests.post(token_url, data=data, headers={'Content-Type': 'application/x-www-form-urlencoded'})
    if response.status_code == 200:
        return response.json().get('access_token'), '', 200
    else:
        return None, 'Error fetching access token: ' + str(response.content), response.status_code

def fetch_linkedin_user_info(access_token):
    userinfo_url = 'https://api.linkedin.com/v2/userinfo'
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    response = requests.get(userinfo_url, headers=headers)
    if response.status_code == 200:
        user_info = response.json()

        # Extracting the 'sub' value from the user info dictionary
        sub_value = user_info.get('sub')

        return user_info, sub_value
    else:
        return None, None

def post_to_linkedin(text, sub_value, access_token, image_urn=None):
    post_url = 'https://api.linkedin.com/v2/ugcPosts'

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "author": f"urn:li:person:{sub_value}",
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {
                    "text": text
                },
                "shareMediaCategory": "NONE"
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
        }
    }
    
    if image_urn:  # Check if an image URN is provided
        payload["specificContent"]["com.linkedin.ugc.ShareContent"]["media"] = [
            {
                "status": "READY",
                "description": {
                    "text": text
                },
                "media": f"{image_urn}",  # Add Media entry with the URN
                "title": {
                    "text": "Post Title"
                }
            }
        ]
        payload["specificContent"]["com.linkedin.ugc.ShareContent"]["shareMediaCategory"] = "IMAGE"  # Set category to IMAGE

    response = requests.post(post_url, json=payload, headers=headers)

    if response.status_code == 201:
        return response.json()
    else:
        return f'Error posting to LinkedIn: {response.content}', response.status_code


def upload_image_to_linkedin(image_file_path, sub_value, access_token):
    register_upload_url = 'https://api.linkedin.com/v2/assets?action=registerUpload'
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
        'X-Restli-Protocol-Version': '2.0.0'  # Necessary header for image uploads
    }
    
    register_upload_payload = {
        "registerUploadRequest": {
            "recipes": [
                "urn:li:digitalmediaRecipe:feedshare-image"
            ],
            "owner": f'urn:li:person:{sub_value}',
            "serviceRelationships": [{
                "relationshipType": "OWNER",
                "identifier": "urn:li:userGeneratedContent"
            }]
        }
    }
    
    register_response = requests.post(
        register_upload_url, json=register_upload_payload, headers=headers)
    
    if register_response.status_code != 200:
        return f'Error registering image upload: {register_response.content}', register_response.status_code
    
    # Extract upload URL and asset URN if registration was successful
    upload_url = (register_response.json()
                  .get('value')
                  .get('uploadMechanism')
                  .get('com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest')
                  .get('uploadUrl'))
    asset = register_response.json().get('value').get('asset')

    # Upload the image using the 'put' method and the received 'upload_url'
    headers_for_upload = {'Authorization': f'Bearer {access_token}'}
    
    with open(image_file_path, 'rb') as image_file:
        image_data = image_file.read()
    
    upload_response = requests.put(upload_url, data=image_data, headers=headers_for_upload)

    # Change this check to accept both 200 (OK) and 201 (Created) as successful status codes
    if upload_response.status_code in (200, 201):
        # Clean up the temporary file after upload
        os.remove(image_file_path)
        
        # Return the asset which contains the URN on success
        return asset
    else:
        # Handle an actual error
        print(f'Error status code: {upload_response.status_code}')
        print(f'Response content: {upload_response.content}')
        return f'Error status code: {upload_response.status_code}', upload_response.status_code
    