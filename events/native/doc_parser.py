from __future__ import division
import re

# text
def clean_text(item):
    item = re.sub(r'[\'"|\n\t,.:;()\-\/]+', ' ', item.encode('ascii', 'ignore').strip()).lower()
    item = re.sub(r'\s+', ' ', item)
    return item

def clean_texts(items):
    items = [clean_text(item) for item in items]
    return items

# image source 
def clean_image_source(src):
    """data:image/gif;base64,R0lGODdh"""
    if src and src.startswith('data:image/'):
        src = src[:(src.find(';base64,'))]
    return src

# script/style/image text stats
def text_stats(items):
    text_lengths = [len(item) for item in items]
    big_text_lengths = [text_len for text_len in text_lengths if text_len > 2]
    if len(text_lengths) > 0:
        avg = round(sum(text_lengths)/len(text_lengths), 1)
    else:
        avg = 0.0
    if len(big_text_lengths) > 0:
        b_avg = round(sum(big_text_lengths)/len(big_text_lengths), 1)
    else:
        b_avg = 0.0
    return len(text_lengths), avg, len(big_text_lengths), b_avg

# individual tags
def get_misc_counts(soup):
    return {
        "div": len(soup.find_all('div')),
        "input": len(soup.find_all('input')),
        "button": len(soup.find_all('button')),
        "meta": len(soup.find_all('meta')),
        "link": len(soup.find_all('link'))
        }

def get_paragraphs(soup):
    paragraphs = soup.find_all('p')
    cleaned_texts = clean_texts([p.text for p in paragraphs])
    return cleaned_texts 

def get_image_data(soup):
    items = soup.find_all('img')
    srcs = [clean_image_source(image.get('src')) for image in items]
    cnt, avg, b_cnt, b_avg = text_stats([item.text for item in items])
    return {
        "srcs": srcs, 
        "cnt": cnt,
        "avg": avg,
        "b_cnt": b_cnt,
        "b_avg": b_avg
        }

def get_script_data(soup):
    items = soup.find_all('script')
    srcs = [script.get('src') for script in items]
    cnt, avg, b_cnt, b_avg = text_stats([item.text for item in items])
    return {
        "srcs": srcs, 
        "cnt": cnt,
        "avg": avg,
        "b_cnt": b_cnt,
        "b_avg": b_avg
        }

def get_style_data(soup):
    items = soup.find_all('style')
    srcs = [style.get('src') for style in items]
    cnt, avg, b_cnt, b_avg = text_stats([item.text for item in items])
    return {
        "srcs": srcs, 
        "cnt": cnt,
        "avg": avg,
        "b_cnt": b_cnt,
        "b_avg": b_avg
        }

def get_title(soup):
    title = soup.find('title')
    title = 'e' if not title else title.text
    title = 'longtitle' if len(title) > 300 else clean_text(title)
    return title

def get_links(soup):
    hrefs = []
    texts = []
    links = soup.find_all('a', href=True)
    hrefs = [a['href'] for a in links]
    texts = clean_texts([a.text for a in links])
    return hrefs, texts

# the whole document
def parse(soup, filename):
    title = get_title(soup)
    pars = get_paragraphs(soup)

    ahrefs, atexts = get_links(soup)

    image_data = get_image_data(soup)
    script_data = get_script_data(soup)
    style_data = get_style_data(soup)

    misc_counts = get_misc_counts(soup)

    doc = {
        "id": filename, 

        "title": title,
        "par": pars,

        "ahref": ahrefs,
        "atext": atexts,

        "img": image_data,
        "script": script_data,
        "style": style_data,

        "misc": misc_counts
        }
    return doc

