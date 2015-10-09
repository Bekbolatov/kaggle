import pandas as pd

top1m = pd.read_csv('data/top-1m.csv', names=['rank', 'domain'], index_col='domain')

def get_rank(domain):
    try:
        return top1m.loc[domain][0]
    except KeyError:
        return 1000000
    
