from __future__ import division
import re
from common_domains import soup2link_profile

features = """
1) 'content="width=device.+?"viewport"'
2) '"viewport"'
3) 'fonts.googleapis'
4)  features based on a href URL domain names
"""

pattern_1 = re.compile('content="width=device.+?"viewport"')
pattern_2 = re.compile('"viewport"')
pattern_3 = re.compile('fonts.googleapis')

pattern_1_3 = [pattern_1, pattern_2, pattern_3]

def parse(soup, text, filename):
    p = [1 if p.search(text) else 0 for p in pattern_1_3]
    
    links_profile = soup2link_profile(soup)

    values = {
        "id": filename, 
        "viewport": p,
        "links_profile": links_profile
        } 

    return values
