import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from google_secret_utils import get_secret_version
import os

# Define the project ID and the secret IDs for username and app password
PROJECT_ID = 'kumori-404602'
GMAIL_USERNAME_SECRET_ID = 'KUMORI_GMAIL_USERNAME'
GMAIL_APP_PASSWORD_SECRET_ID = 'KUMORI_GMAIL_APP_PASSWORD'

# Retrieve the Gmail credentials from the Google Secret Manager
GMAIL_USER = get_secret_version(PROJECT_ID, GMAIL_USERNAME_SECRET_ID)
GMAIL_PASSWORD = get_secret_version(PROJECT_ID, GMAIL_APP_PASSWORD_SECRET_ID)

# Function to send emails
def send_email(subject, body, to_emails, attachment_paths=None):
    # Setup email headers and recipients
    message = MIMEMultipart()
    message['From'] = 'Kumori.ai <{}>'.format(GMAIL_USER)
    message['To'] = ', '.join(to_emails)
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))
    
    # Process attachments if any
    if attachment_paths:
        for attachment_path in attachment_paths:
            part = MIMEBase('application', 'octet-stream')
            with open(attachment_path, 'rb') as file:
                part.set_payload(file.read())
            encoders.encode_base64(part)
            # Only pass the basename of the file to the filename parameter
            part.add_header(
                'Content-Disposition',
                'attachment',
                filename=os.path.basename(attachment_path)
            )
            message.attach(part)
        
    # Connect to Gmail SMTP server and send email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.set_debuglevel(1)  # Enable debug output to console
        server.login(GMAIL_USER, GMAIL_PASSWORD)
        server.send_message(message)
        print('Email sent successfully')

# Function to create a sample text file
def create_sample_text_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)
    print(f"Created file {filename} with content: {content}")

# Test the function
if __name__ == '__main__':
    # Create sample text file
    filename = 'sample.txt'
    content = 'hello_world'
    create_sample_text_file(filename, content)

    # Sample usage
    subject = 'Test Email with Attachment from Python'
    body = 'This email contains an attachment sent from the python email utility.'
    to_emails = ['email@andy.com']

    # Get the full path to the attachment
    attachment_path = os.path.join(os.getcwd(), filename)
    attachment_paths = [attachment_path]

    send_email(subject, body, to_emails, attachment_paths)