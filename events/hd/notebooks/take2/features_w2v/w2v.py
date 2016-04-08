import os
import pandas as pd
import re


package_directory = os.path.dirname(os.path.abspath(__file__))
base_loc = package_directory
loc = '../../%s'
path_feat_tfidf = 'FEAT_W2V.df'

def create_features():
    import logging as log
    log.basicConfig(format='[%(asctime)s] %(message)s', level=log.INFO)
    log.info("W2V features generation")

    queries = pd.read_pickle(base_loc + '/' + loc % 'FEATURES_WITH_TEXT_1.data')
    log.info('read "queries", shape: %s' % str(queries.shape))

    log.info('\t load dictionary')

    log.info('\t process data')

    feat1 = []
    for i, row in queries.iterrows():
        if i%10000 == 0:
            log.info("... %d" % i)
        feat1.append(calc_feat1(row))


def calc_feat1(row):
    q = row['query']
    #, row['product_title']
    return len(q)

