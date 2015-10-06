from __future__ import division
import re

features = """
1) "'set', 'forseSSL', true"
2) "'send', 'pageview'"
3) "WordPress"
4) "/wp-includes/js/jquery/"
5) modernizr
6) "sponsored" : lower()

7) "facebook.+;appId=(\d+)" : default 0

8) ---  Google Analytics ---
   "UA-(\d{5,10})-(\d+)"    :
   first matching line: 
        + id (0),
        + length (1000), 
        + number(total number of lines)
    +count of matches > 2
"""

pattern_1 = re.compile("'set', 'forseSSL', true")
pattern_2 = re.compile("'send', 'pageview'")
pattern_3 = re.compile("WordPress")
pattern_4 = re.compile("/wp-includes/js/jquery/")
pattern_5 = re.compile("modernizr")
pattern_6 = re.compile("[sS]ponsored")

pattern_1_6 = [pattern_1, pattern_2, pattern_3, pattern_4, pattern_5, pattern_6]

pattern_fb = re.compile("facebook.+;appId=(\d+)")
pattern_ga = re.compile("UA-(\d{5,10})-(\d+)")

#tokens = word_splitter.findall(dirty_text.lower())
#p.search(s)    # The result of this is referenced by variable name '_'
#_.group(1)     # group(1) will return the 1st capture.



def parse(soup, text, filename):
    p = [1 if p.search(text) else 0 for p in pattern_1_6]

    fb_id = 0
    m = pattern_fb.search(text)
    if m:
        fb_id = m.group(1)
   
    ga_id = 0
    ms = pattern_ga.findall(text)
    if len(ms) > 0:
        ga_id = ms[0][0]
        ga_subid = ms[0][1]

    ga_len = 1000
    ga_line = 5000

    for n, line in enumerate(text):
        if pattern_ga.search(line):
            ga_len = len(line)
            ga_line = n
            break

    values = {
        "id": filename, 
        "basic": p,
        "fb_id": fb_id,
        "ga_id": ga_id,
        "ga_subid": ga_subid,
        "ga_length": ga_len,
        "ga_line": ga_line,
        } 

    return values
