import re
from urlparse import urlparse
from common_domain_names import get_rank
from collections import Counter

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
        return 'bad_domain'

def ahref2text(ahref):
    try:
        parsed_url = urlparse(ahref)
        raw_domain = parsed_url.netloc
        domain = clean_domain(raw_domain)
        if domain == '':
            return 'no_domain'
        return domain
    except Exception as inst:
        print inst
        return 'bad_domain'

def a2domain(a):
    try:
        ahref = a['href']
        return ahref2text(ahref)
    except Exception as inst:
        print inst
        return 'bad_domain'

def soup2link_profile(soup):
    domains = Counter([a2domain(a) for a in soup.find_all('a')])
    profile = {}

    profile['domain_set_size'] = len(domains)
    profile['no_domain'] = domains['no_domain']
    profile['bad_domain'] = domains['bad_domain']
    
    domains = [ (k, v) for k,v in domains.items() if k not in set(['no_domain', 'common_domain', 'bad_domain'])]
    domains = sorted(domains, key=lambda p: -p[1])
    domains = [ (domain, count, get_rank(domain)) for (domain, count) in domains]

    unpopular_domains = [(domain, count, rank) for (domain, count, rank) in domains if rank > 2000 and count > 10]
    if len(unpopular_domains) > 0:
        profile['candidate_site'] = unpopular_domains[0][0]
        profile['candidate_rank'] = unpopular_domains[0][2]
    else:
        profile['candidate_site'] = None
        profile['candidate_rank'] = 0
     
    if len(domains) >= 2:
        profile['candidate_ratio'] = 1.0 * domains[0][1] / domains[1][1]
    else:
        profile['candidate_ratio'] = 1.0

    if len(unpopular_domains) >= 1:
        profile['total_ratio'] = 1.0 * unpopular_domains[0][1] / sum([count for (domain, count, rank) in unpopular_domains])
    else:
        profile['total_ratio'] = 0.0

    if len(domains) >= 2 and domains[0][1] > domains[1][1] * 3 and domains[0][1] > 10:
        profile['candidate_score'] = domains[0][2]  
    else:
        profile['candidate_score'] = 0

    profile['domains'] = domains
    return profile 





# DEBUG
from bs4 import BeautifulSoup as bs
def soy(fna):
    try:
        with open('/Users/rbekbolatov/data/kaggle/native/sample/two/positives/' + fna) as f:
            return bs(f)
    except Exception:
        with open('/Users/rbekbolatov/data/kaggle/native/sample/two/negatives/' + fna) as f:
            return bs(f)
    
