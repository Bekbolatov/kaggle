from urlparse import urlparse
import re
import collections

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

foreign_domain_pattern = re.compile(r'^[^\.]{2,3}\.[^\.]{2}$')

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

def a2text(ahref, atext):
    try:
        ahref = a['href']
        atext = a.text.encode('ascii', 'ignore')
        parsed_url = urlparse(ahref)
        raw_domain = parsed_url.netloc
        if raw_domain in social_domain_names:
            return ''
        raw_path = parsed_url.path
        domain = clean_domain(raw_domain)
        text = clean_text(raw_path + ' ' + atext)
        return domain + ' ' + text
    except Exception as inst:
        return 'bad_bad_domain'
