import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

package_directory = os.path.dirname(os.path.abspath(__file__))
base_loc = package_directory
loc = '../../%s'
path_feat_tfidf = 'FEAT_TFIDF.df'

simple_word = re.compile('([0-9]|units|xby)')

def create_features():
    import logging as log
    log.basicConfig(format='[%(asctime)s] %(message)s', level=log.INFO)
    log.info("TF-IDF features generation")

    queries = pd.read_pickle(base_loc + '/' + loc % 'FEATURES_WITH_TEXT_1.data')
    log.info('read "queries", shape: %s' % str(queries.shape))

    # vectorizers for similarities
    log.info('\t fit vectorizers')

    tfv_title = TfidfVectorizer(ngram_range=(1,2), min_df=4)
    tfv_title_desc = TfidfVectorizer(ngram_range=(1,2), min_df=4)
    tfv_desc = TfidfVectorizer(ngram_range=(1,2), min_df=4)
    tfv_all = TfidfVectorizer(ngram_range=(1,2), min_df=4)

    log.info('\t ... query - title')
    tfv_title.fit(
        list(queries['query'].values) +
        list(queries['product_title'].values)
    )
    log.info('\t ... query - description')
    tfv_desc.fit(
        list(queries['query'].values) +
        list(queries['product_description'].values)
    )
    log.info('\t ... query - title,description')
    tfv_title_desc.fit(
        list(queries['query'].values) +
        list(queries['product_title'].values) +
        list(queries['product_description'].values)
    )
    log.info('\t ... query - title,description,attrs')
    tfv_all.fit(
        list(queries['query'].values) +
        list(queries['product_title'].values) +
        list(queries['product_description'].values) +
        list(queries['attrs'].values)
    )

    # for train
    log.info('\t process dataset')
    cosine_title = []
    cosine_desc = []
    cosine_title_desc = []
    cosine_all = []
    set_title = []
    set_desc = []
    set_attr = []
    for i, row in queries.iterrows():
        if i%1000 == 0:
            log.info("... %d" % i)
        cosine_title.append(calc_cosine_dist(row['query'], row['product_title'], tfv_title))
        cosine_desc.append(calc_cosine_dist(row['query'], row['product_description'], tfv_desc))
        cosine_title_desc.append(calc_cosine_dist(row['query'], row['product_title'] + ' ' + row['product_description'], tfv_title_desc))
        cosine_all.append(calc_cosine_dist(row['query'], row['product_title'] + ' ' + row['product_description'] + row['attrs'], tfv_all))

        set_title.append(calc_set_intersection(row['query'], row['product_title']))
        set_desc.append(calc_set_intersection(row['query'], row['product_description']))
        set_attr.append(calc_set_intersection(row['query'], row['attrs']))


    feats_df = queries['query'].copy()

    feats_df['cosine_title'] = cosine_title
    feats_df['cosine_desc'] = cosine_desc
    feats_df['cosine_title_desc'] = cosine_title_desc
    feats_df['cosine_all'] = cosine_all
    feats_df['set_title'] = set_title
    feats_df['set_desc'] = set_desc
    feats_df['set_attr'] = set_attr

    feats_df.drop('query', axis=1)
    pd.to_pickle(feats_df, base_loc + '/' + path_feat_tfidf)


def calc_cosine_dist(text_a ,text_b, vect):
    return pairwise_distances(vect.transform([text_a]), vect.transform([text_b]), metric='cosine')[0][0]


def calc_set_intersection(text_a, text_b):
    a = set([x for x in text_a.split() if not simple_word.search(x)])
    b = set(text_b.split())
    return len(a.intersection(b)) *1.0 / (1 + len(a))

