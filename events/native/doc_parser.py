

def parse(soup, filename):
    title = soup.title.text
    return {"id": filename, "title": title} 

