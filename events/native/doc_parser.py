from __future__ import division
import re
from urlparse import urlparse
import collections
from word_stemmer import stem
import word_lemmatizer 
from word_lemmatizer import sentence2vec, words

foreign_domain_pattern = re.compile(r'^[^\.]{2,3}\.[^\.]{2}$')
word_splitter = re.compile(r'[a-z]{4,}')

def clean_text(dirty_text):
    if not dirty_text:
        return ''
    if type(dirty_text) != str:
        dirty_text = ' '.join(dirty_text)
    tokens = word_splitter.findall(dirty_text.lower())
    tokens = [stem(token) for token in tokens]
    clean_text = ' '.join(tokens)
    return clean_text

social_domain_names = [
    'twitter.com',
    'facebook.com',
    'google.com',
    'pinterest.com',
    'linkedin.com',
    'reddit.com',
    'olark.com',
    'disqus.com',
    'apple.com',
    'yelp.com',
    'imgur.com',
    'instagram.com',
    'youtube.com',
    'tumblr.com',
    'feedburner.com',
    ]

def clean_domain(host):
    try:
        num_pieces = host.count('.') + 1
        if num_pieces <= 2:
            return host
        shortened_host = host[(host.find('.')+1):]
        if num_pieces == 3: 
            if foreign_domain_pattern.match(shortened_host):
                return host
            else:
                return shortened_host
        more_shortened_host = shortened_host[(shortened_host.find('.')+1):]
        if foreign_domain_pattern.match(more_shortened_host):
            return shortened_host
        else:
            return more_shortened_host        
    except Exception as inst:
        print inst
        print href
        return 'bad_bad_domain'

def a2text(a):
    try:
        ahref = a['href']
        atext = a.text.encode('ascii', 'ignore')
        parsed_url = urlparse(ahref)
        raw_domain = parsed_url.netloc
        if raw_domain in social_domain_names or raw_domain == '':
            return ''
        raw_path = parsed_url.path
        #domain = clean_domain(raw_domain)
        text = raw_path + ' ' + atext
        text = clean_text(text)
        #text = domain + ' ' + clean_text(text)
        text = re.sub(r'\s+', ' ', text)
        return text
    except Exception as inst:
        print inst
        return 'bad_bad_domain'


def parse(soup, text, filename):
    text_p = ' '.join([item.text for item in soup.find_all('p')])
    tag_title = ' '.join([item.text for item in soup.find_all('title')])

    text = text_p + ' ' + tag_title

    values = {
        "id": filename, 
        "words": words(text),
        "word2vec": sentence2vec(text).tolist()
        } 

    return values
