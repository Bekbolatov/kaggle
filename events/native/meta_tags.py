
def clean_meta(meta):
    try:
        return meta['name']
        
    except Exception:
        try:
            return meta['property']
        except Exception:
            return None
        return None

def get_metas(soup):
    metas = [clean_meta(m) for m in soup.find_all('meta')]
    return [meta for meta in metas if meta]


