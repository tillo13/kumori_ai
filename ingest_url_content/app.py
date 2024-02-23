import requests
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup
import re

def check_robots(url):
    rp = RobotFileParser()
    parsed_url = urlparse(url)
    rp.set_url("{0.scheme}://{0.netloc}/robots.txt".format(parsed_url))
    rp.read()
    can_fetch = rp.can_fetch("*", url)
    return can_fetch

def main():
    url = input('Enter the URL to scrape: ')        
    elab_msg = '' # Detailed message to user.
    try:
        response = requests.head(url, timeout=5) # issue a HEAD request to get http headers
        if 'text/html' in response.headers['Content-Type']: 
            if response.status_code == 200:
                if check_robots(url):
                    print('Yes, I can get there.')
                    choice = input('Do you want me to print the HTML? 1. Yes 2. No \n')
                    if choice == '1':
                        response = requests.get(url)
                        html_content = response.text
                        print(html_content)
                        char_count = len(html_content)
                        print(f'The page has {char_count} characters.')
                        
                        # clean up the HTML and javascript
                        soup = BeautifulSoup(html_content, "html.parser")
                        for script in soup(["script", "style"]): 
                            script.extract()

                        clean_text = soup.get_text()
                        clean_text = re.sub(r'\s+', ' ', clean_text)   # condense multiple spaces into one
                        clean_char_count = len(clean_text)

                        print(f'After cleaning, the HTML has {clean_char_count} characters.')

                        choice = input('Do you want to see the cleaned code? 1. Yes 2. No \n')
                        if choice == '1':
                            print(clean_text)
                    return
                else:
                    elab_msg = '... because access is blocked by robots.txt'
            else:
                elab_msg = '... because the server responded with status code: {0}'.format(response.status_code)
        else:
            elab_msg = "... because the URL does not appear to point to an HTML document (Content-Type: {0})".format(response.headers['Content-Type'])
    except Exception as e:
        elab_msg = '... because: {0}'.format(str(e))
    
    print('No, I cannot get there {0}'.format(elab_msg))

if __name__ == "__main__":
    main()