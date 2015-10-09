import pandas as pd

top1m = pd.read_csv('/home/ec2-user/repos/bekbolatov/kaggle/events/native/data/top-1m.dat', names=['rank', 'domain'], index_col='domain')

def get_rank(domain):
    try:
        return top1m.loc[domain][0]
    except KeyError:
        return 1000000
    
