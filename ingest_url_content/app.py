from flask import Flask, request, render_template_string
import requests
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup
import re
import html

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template_string('''
    <html>
        <head>
            <title>URL Scraper</title>
        </head>
        <body>
            <h1>Enter a URL to Scrape:</h1>
            <form action="/scrape" method="GET">
                <input type="text" name="url" />
                <input type="submit" value="Submit" />
            </form>
            <hr />
            <h1>Or enter HTML to clean:</h1>
            <form action="/clean" method="POST">
                <textarea name="html_content" rows="20" cols="80"></textarea>
                <input type="submit" value="Submit" />
            </form>
            <a href="/">Again?</a>
        </body>
    </html>
    ''')

@app.route('/clean', methods=['POST'])
def clean():
    html_content = request.form.get('html_content')
    char_count = len(html_content)

    soup = BeautifulSoup(html_content, "html.parser")
    [s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title'])]
    visible_text = soup.getText()
    cleaned_html = re.sub('\s+', ' ', visible_text)
    clean_char_count = len(cleaned_html)
    escaped_cleaned_html = html.escape(cleaned_html)

    return render_template_string('''
    <html>
        <head>
            <title>HTML Cleaner</title>
        </head>
        <body>
            <a href="/">Again?</a>
            <h1>The HTML had {{char_count}} characters.</h1>
            <h1>After cleaning, the HTML has {{clean_char_count}} characters.</h1>
                        <textarea readonly rows="20" cols="80">{{escaped_cleaned_html}}</textarea>
        </body>
    </html>
    ''', char_count=char_count, clean_char_count=clean_char_count, escaped_cleaned_html=escaped_cleaned_html)

@app.route('/scrape', methods=['GET'])
def scrape():
    url = request.args.get('url')
    full_html = ''
    cleaned_html = ''
    char_count = 0
    clean_char_count = 0
    error_message = ''
    try:
        response = requests.head(url, timeout=5)
        if 'text/html' in response.headers['Content-Type']:
            if response.status_code == 200:
                if check_robots(url):
                    full_html = requests.get(url).text
                    char_count = len(full_html)

                    soup = BeautifulSoup(full_html, "html.parser")
                    [s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title'])]
                    visible_text = soup.getText()
                    cleaned_html = re.sub('\s+', ' ', visible_text)
                    clean_char_count = len(cleaned_html)
                    escaped_full_html = html.escape(full_html)
                    escaped_cleaned_html = html.escape(cleaned_html)

                    return render_template_string('''
                    <html>
                        <head>
                                                  <a href="/">Again?</a>
                            <title>URL Scraper</title>
                            <style>
                                textarea {
                                    width: 100%;
                                    height: 200px;
                                }
                            </style>
                        </head>
                        <body>
                            
                            <h2>User submitted this URL: {{url}}</h2>
                            <h1>The page has {{char_count}} characters.</h1>
                            <textarea readonly>{{escaped_full_html}}</textarea>
                            <h1>After cleaning, the HTML has {{clean_char_count}} characters.</h1>
                            <textarea readonly>{{escaped_cleaned_html}}</textarea>
                        </body>
                    </html>
                    ''', char_count=char_count, clean_char_count=clean_char_count, escaped_full_html=escaped_full_html, escaped_cleaned_html=escaped_cleaned_html, url=url)
                else:
                    error_message = 'Access is blocked by robots.txt'
            else:
                error_message = 'The server responded with status code: {}'.format(response.status_code)
        else:
            error_message = 'The URL does not point to an HTML document (Content-Type: {})'.format(response.headers['Content-Type'])
    except Exception as e:
        error_message = 'Error occurred: {}'.format(str(e))

    if error_message:
        return render_template_string('''
        <html>
            <head>
                <title>Error</title>
            </head>
            <body>
                <h1>Error when processing URL: {{url}}</h1>
                <h2>Error details: {{error_message}}</h2>
                <a href="/">Again?</a>
            </body>
        </html>
        ''', error_message=error_message, url=url)

@app.route('/show_code', methods=['POST'])
def show_code():
    escaped_cleaned_html = request.form.get('escaped_cleaned_html')
    return render_template_string('''
    <html>
        <head>
            <title>Cleaned Code</title>
        </head>
        <body>
            <a href="/">Again?</a>
            <h1>Cleaned Code:</h1>
            <pre><code>{{escaped_cleaned_html}}</code></pre>
        </body>
    </html>
    ''', escaped_cleaned_html=escaped_cleaned_html)

def check_robots(url):
    rp = RobotFileParser()
    parsed_url = urlparse(url)
    rp.set_url("{0.scheme}://{0.netloc}/robots.txt".format(parsed_url))
    rp.read()
    can_fetch = rp.can_fetch("*", url)
    return can_fetch

def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    # remove script and style tags
    for script in soup(["script", "style"]):
        script.extract()
    # convert to utf-8
    text = soup.prettify("utf-8")
    # convert to text string
    text = str(text, 'utf-8')

    # remove HTML tags using regex
    text = re.sub('<.*?>', '', text)

    # remove blank lines and extra line breaks
    text = re.sub(r'\s+', ' ', text)
        
    return text

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)