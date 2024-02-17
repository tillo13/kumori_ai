import os.path
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

from requests import Request as HttpRequest

import json

import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# If modifying these SCOPES, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def upload_file(file_path):
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())  # This should now be the correct Request from google.auth.transport.requests
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)

    MIME_TYPE_MAP = {
    '.jpeg': 'image/jpeg',
    '.jpg': 'image/jpeg',
    '.png': 'image/png',
}

    file_name = os.path.basename(file_path)
    mime_type = MIME_TYPE_MAP.get(os.path.splitext(file_name)[1].lower(), "application/octet-stream")  # new added line for MIME type
    folder_id = '18eVZDUPlpRosmHoqk997Tn8nqs0SRD1o'  # Include the folder ID here
    file_metadata = {'name': file_name, 'parents': [folder_id]}
    media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)  # mime_type is now dynamically set


    # Send file to Google Drive
    print("Sending this to Google Drive: ")
    print(json.dumps(file_metadata, indent=4))
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print("Receiving this from Google Drive: ")
    print(json.dumps(file, indent=4))
    
    if file:
        file_id = file.get('id', 'File ID not found')
        
        permission = {
            'type': 'anyone',
            'role': 'reader',
        }

        # Create permission for the file
        print("Creating permission for the file with these data: ")
        print(json.dumps(permission, indent=4))
        perm_result = service.permissions().create(fileId=file_id, body=permission).execute()

        # Create a Direct Download Link
        direct_link = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        print("Sharable link created: ", direct_link)
        return direct_link

    print("File Upload Failed.")
    return None

if __name__ == '__main__':
    upload_file()