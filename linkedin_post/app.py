from flask import Flask, request, render_template, redirect
from utils import linkedin_utils

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('get_linkedin_access_token.html')

@app.route('/login_to_linkedin')
def login():
    return linkedin_utils.redirect_to_linkedin_auth_url()

@app.route('/linkedin/callback')
def callback():
    result = linkedin_utils.process_linkedin_callback(request.args)
    if 'error' in result:
        return result['error'], result['status_code']
    return render_template('post_to_linkedin.html', **result['data'])

@app.route('/post_to_linkedin', methods=['POST'])
def post_to_linkedin():
    result = linkedin_utils.post_to_linkedin_with_image(request)
    if 'error' in result:
        return result['error'], result['status_code']
    return result['message']

if __name__ == '__main__':
    app.run(debug=True, port=5000)