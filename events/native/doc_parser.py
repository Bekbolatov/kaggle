from __future__ import division
import re
from urlparse import urlparse
import collections
from word_stemmer import stem

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
        #text = domain + ' ' + clean_text(text)
        text = re.sub(r'\s+', ' ', text)
        return text
    except Exception as inst:
        print inst
        return 'bad_bad_domain'


def parse(soup, filename):
    tag_p = clean_text([item.text.encode('ascii', 'ignore') for item in soup.find_all('p')])
    tag_title = clean_text([item.text.encode('ascii', 'ignore') for item in soup.find_all('title')])
    tag_meta = clean_text([item.description.encode('ascii', 'ignore') for item in soup.find_all('meta') if item.description])
    tag_a = ' '.join([a2text(item) for item in soup.find_all('a', href=True)])

    text = ' '.join([tag_p, tag_title, tag_meta, tag_a])

    return {
        "id": filename, 
        "text": text,
        "meta": tag_meta,
        }

