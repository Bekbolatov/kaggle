import re


def get_paragraphs(soup):
    paragraphs = soup.findAll('p')
    cleaned_texts = [re.sub(r'[\'"|\n\t,.:;()\-\/]+', ' ', p.text.encode('ascii', 'ignore').strip()).lower() for p in paragraphs]
    return cleaned_texts 

def get_title(soup):
    title = soup.find('title')
    title = '' if not title else title.text
    title = 'long_title' if len(title) > 200 else title.encode('ascii', 'ignore').strip().replace('\n',' ')
    return title

def get_links(soup):
    hrefs = []
    texts = []
    links = soup.findAll('a', href=True)
    hrefs = [a.href for a in links]
    texts = [re.sub(r'[\'"|\n\t,.:;()\-\/]+', ' ', a.text.encode('ascii', 'ignore').strip()) for a in links]
    return hrefs, texts


def parse(soup, filename):
    title = get_title(soup)
    pars = get_paragraphs(soup)
    ahrefs, atexts = get_links(soup)
    doc = {
        "id": filename, 
        "title": title,
        "pars": pars,
        "ahrefs": ahrefs,
        "atexts": atexts
        }
    return doc

