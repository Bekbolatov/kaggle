from __future__ import division
import re
from parsing_url import a2text

word_splitter = re.compile(r'[a-z]{4,}')

def clean_text(dirty_text):
    if not dirty_text:
        return ''
    if type(dirty_text) != str:
        dirty_text = ' '.join(dirty_text)
    tokens = word_splitter.findall(dirty_text.lower())
    clean_text = ' '.join(tokens)
    return clean_text

def parse(soup, filename):
    tag_p = [item.text.encode('ascii', 'ignore') for item in soup.find_all('p')]
    tag_title = [item.text.encode('ascii', 'ignore') for item in soup.find_all('title')]
    tag_a = [a2text(item) for item in soup.find_all('a', href=True)]

    text = clean_text(tag_p + tag_title + tag_a)

    return {
        "id": filename, 
        "text": text
        }

