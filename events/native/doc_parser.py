import re


def get_paragraphs(soup):
    paragraphs = soup.find_all('p')
    cleaned_texts = [re.sub(r'[\'"|\n\t,.:;()\-\/]+', ' ', p.text.encode('ascii', 'ignore').strip()).lower() for p in paragraphs]
    return cleaned_texts 

def get_images(soup):
    images = soup.find_all('img')
    srcs = [image.get('src') for image in images]
    return srcs 

def get_scripts(soup):
    scripts = soup.find_all('script')
    srcs = [script.get('src') for script in scripts]
    return srcs 

def get_styles(soup):
    styles = soup.find_all('style')
    srcs = [style.get('src') for style in styles]
    return srcs 

def get_title(soup):
    title = soup.find('title')
    title = '' if not title else title.text
    title = 'longtitle' if len(title) > 200 else title.encode('ascii', 'ignore').strip().replace('\n',' ')
    return title

def get_links(soup):
    hrefs = []
    texts = []
    links = soup.findAll('a', href=True)
    hrefs = [a['href'] for a in links]
    texts = [re.sub(r'[\'"|\n\t,.:;()\-\/]+', ' ', a.text.encode('ascii', 'ignore').strip()) for a in links]
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

