import numpy as np
import pandas as pd

#LOC = '/home/ec2-user/data/hd/unpacked/'
LOC = '/Users/rbekbolatov/data/kaggle/homedepot/'
data_loc = 'data/%s.df'


def create_combined():
    _queries = pd.read_pickle(data_loc % 'QUERIES')
    _products = pd.read_pickle(data_loc % 'PRODUCTS')
    _product_queries = pd.read_pickle(data_loc % 'PRODUCT_QUERIES')
    train_labels = pd.read_pickle(data_loc % 'LABELS_TRAIN')
    test_labels = pd.read_pickle(data_loc % 'LABELS_TEST')

    df_products = pd.concat([_product_queries, _products], axis=1)
    df_products.to_pickle(data_loc % 'COMBINED_PRODUCT')


def get_combined(loc=LOC):
    df_products = pd.read_pickle(data_loc % 'COMBINED_PRODUCT')
    return df_products


def re_clean_tokenize_text():
    df_products =  get_combined()
    import text_parsing
    reload(text_parsing)
    from text_parsing import str_stem

    df_products['product_title'] = df_products['product_title'].apply(str_stem)
    df_products['product_description'] = df_products['product_description'].apply(str_stem)
    df_products['attributes'] = df_products['attributes'].apply( lambda d: {str_stem(k): str_stem(v) for k,v in d.items()} )
    df_products['queries'] = df_products['queries'].apply( lambda d: {k: str_stem(v) for k,v in d.items()} )
    df_products.to_pickle(data_loc % 'COMBINED_PRODUCT_TOKENIZED')

def get_combined_clean_tokenized(loc=LOC):
    df_products = pd.read_pickle(data_loc % 'COMBINED_PRODUCT_TOKENIZED')
    return df_products


import google_spell
reload(google_spell)
from google_spell import correct_spelling

def correct_spell(w):
    v = correct_spelling(w)
    if w == v:
        return v + " sxdli"
    else:
        return v

def clean_text():
    df_products =  get_combined()
    import text_parsing
    reload(text_parsing)
    from text_parsing import str_stem


    df_products['product_title'] = df_products['product_title'].apply(str_stem)
    df_products['product_description'] = df_products['product_description'].apply(str_stem)
    df_products['attributes'] = df_products['attributes'].apply( lambda d: {str_stem(k): str_stem(v) for k,v in d.items()} )
    df_products['queries'] = df_products['queries'].apply( lambda d: {k: str_stem(correct_spell(v)) for k,v in d.items()} )
    df_products.to_pickle(data_loc % 'CLEAN_DATA')

def get_clean_text(loc=LOC):
    df_products = pd.read_pickle(data_loc % 'CLEAN_DATA')
    return df_products
