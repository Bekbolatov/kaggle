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

