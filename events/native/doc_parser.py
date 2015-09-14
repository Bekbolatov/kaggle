import re

def clean_text(item):
    item = re.sub(r'[\'"|\n\t,.:;()\-\/]+', ' ', item.encode('ascii', 'ignore').strip()).lower()
    item = re.sub(r'\s+', ' ', item)
    return item

def clean_texts(items):
    items = [clean_text(item) for item in items]
    return items

def get_paragraphs(soup):
    paragraphs = soup.find_all('p')
    cleaned_texts = clean_texts([p.text for p in paragraphs])
    return cleaned_texts 

def get_image_srcs(soup):
    images = soup.find_all('img')
    srcs = [image.get('src') for image in images]
    return srcs 

def get_script_srcs(soup):
    scripts = soup.find_all('script')
    srcs = [script.get('src') for script in scripts]
    return srcs 

def get_style_srcs(soup):
    styles = soup.find_all('style')
    srcs = [style.get('src') for style in styles]
    return srcs 

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


def parse(soup, filename):
    title = get_title(soup)
    pars = get_paragraphs(soup)
    ahrefs, atexts = get_links(soup)
    images = get_images(soup)
    scripts = get_scripts(soup)
    styles = get_styles(soup)
    doc = {
        "id": filename, 
        "title": title,
        "pars": pars,
        "ahrefs": ahrefs,
        "atexts": atexts,
        "images": images,
        "scripts": scripts,
        "styles": styles
        }
    return doc

