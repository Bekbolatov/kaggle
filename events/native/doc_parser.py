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
    if src.startswith('data:image/'):
        src = src[:(src.find(';base64,'))]
    return src

# individual tags
def get_paragraphs(soup):
    paragraphs = soup.find_all('p')
    cleaned_texts = clean_texts([p.text for p in paragraphs])
    return cleaned_texts 

def get_image_srcs(soup):
    items = soup.find_all('img')
    srcs = [clean_image_source(image.get('src')) for image in items]
    src_text_lengths = [len(item.text) for item in items]
    return srcs, src_text_lengths

def get_script_srcs(soup):
    items = soup.find_all('script')
    srcs = [script.get('src') for script in items]
    src_text_lengths = [len(item.text) for item in items]
    return srcs, src_text_lengths 

def get_style_srcs(soup):
    items = soup.find_all('style')
    srcs = [style.get('src') for style in items]
    src_text_lengths = [len(item.text) for item in items]
    return srcs, src_text_lengths 

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
    images, image_text_lengths = get_image_srcs(soup)
    scripts, script_text_lengths = get_script_srcs(soup)
    styles, style_text_lengths = get_style_srcs(soup)
    doc = {
        "id": filename, 
        "title": title,
        "pars": pars,
        "ahrefs": ahrefs,
        "atexts": atexts,
        "images": images,
        "scripts": scripts,
        "styles": styles,
        "images_textlen": image_text_lengths,
        "script_textlen": script_text_lengths,
        "style_textlen": style_text_lengths
        }
    return doc

