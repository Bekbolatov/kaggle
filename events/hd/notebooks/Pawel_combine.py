
# coding: utf-8

# In[1]:

import time
import re
import random
random.seed(2016)

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import pipeline, grid_search
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer


# In[2]:

dtrain = xgb.DMatrix("train.buffer")
dtest = xgb.DMatrix("test.buffer")
evallist  = [(dtrain,'train')]


# In[3]:

loc = '%s'
#loc = '/home/ec2-user/data/hd/features/%s'
a_o = np.load(loc % 'train_data.npy')
b_o = np.load(loc % 'test_data.npy')
a_brand = np.load(loc % 'features_brand_01_train.npy')
b_brand = np.load(loc % 'features_brand_01_test.npy')
a_other = np.load(loc % 'FEATURES_1d_TRAIN.npy')
b_other = np.load(loc % 'FEATURES_1d_TEST.npy')
a_word_feat = np.load(loc % 'SPECIAL_WORDS_FEAT_TRAIN.npy')
b_word_feat = np.load(loc % 'SPECIAL_WORDS_FEAT_TEST.npy')

a_w2vdot = pd.read_pickle(loc % 'W2V_dots_train.df').drop('relevance', axis=1).values
b_w2vdot = pd.read_pickle(loc % 'W2V_dots_test.df').drop('relevance', axis=1).values

a_w2vdist = np.load(loc % 'W2V_dists_train.npz')['arr_0']
b_w2vdist = np.load(loc % 'W2V_dists_test.npz')['arr_0']

a_w2v_el = np.load(loc % 'W2V_vecs_train.npz')['arr_0']
b_w2v_el = np.load(loc % 'W2V_vecs_test.npz')['arr_0']


# In[23]:

# LOCAL CV
aaa = pd.read_pickle(loc % 'WOQTAL_TRAIN_147')
bbb = pd.read_pickle(loc % 'WOQTAL_TEST_147')
aa = aaa.drop('relevance', axis=1).values
bb = bbb.drop('relevance', axis=1).values
a = np.hstack((a_o, a_brand, a_other, a_word_feat, aa, a_w2vdot, a_w2vdist, a_w2v_el[:,:700]))
b = np.hstack((b_o, b_brand, b_other, b_word_feat, bb, b_w2vdot, b_w2vdist, b_w2v_el[:,:700]))


# In[24]:

#FINAL GENERATE
aaa = pd.read_pickle(loc % 'WOQTAL_TRAIN_ALL')
bbb = pd.read_pickle(loc % 'WOQTAL_TEST_ALL')
aa = aaa.drop('relevance', axis=1).values
bb = bbb.drop('relevance', axis=1).values
a_full = np.hstack((a_o, a_brand, a_other, a_word_feat, aa, a_w2vdot, a_w2vdist, a_w2v_el[:,:700]))
b_full = np.hstack((b_o, b_brand, b_other, b_word_feat, bb, b_w2vdot, b_w2vdist, b_w2v_el[:,:700]))


# In[14]:

# Pawel
aaaa = pd.read_pickle(loc % 'Pawel_train.df')
bbbb = pd.read_pickle(loc % 'Pawel_test.df')


# In[20]:

pawel_train = aaaa.drop(["product_title", "search_term", "product_description",
                                               "atr_text", 'brand_text', 'color_text', 'material_text',
                                               'bullet01_text', 'bullet02_text', 'bullet03_text', 'bullet04_text', 'bullet05_text', 'bullet06_text',
                                               'bullet07_text', 'bullet08_text', 'bullet09_text', 'bullet10_text', 'bullet11_text', 'bullet12_text', 
           'relevance', 
                         'product_uid'
                        ], axis=1)
pawel_test = bbbb.drop(["product_title", "search_term", "product_description",
                                               "atr_text", 'brand_text', 'color_text', 'material_text',
                                               'bullet01_text', 'bullet02_text', 'bullet03_text', 'bullet04_text', 'bullet05_text', 'bullet06_text',
                                               'bullet07_text', 'bullet08_text', 'bullet09_text', 'bullet10_text', 'bullet11_text', 'bullet12_text', 
           'product_uid'
                       ], axis=1)


# In[25]:

a = np.hstack([pawel_train.values, a])
b = np.hstack([pawel_test.values, b])

a_full = np.hstack([pawel_train.values, a_full])
b_full = np.hstack([pawel_test.values, b_full])


# In[27]:

a.shape, a_full.shape, b.shape, b_full.shape


# In[28]:

X_train, X_test, y_train, y_test = train_test_split(a, dtrain.get_label(), test_size=0.20, random_state=147) # 0.20, 147
gX_train = xgb.DMatrix(data=X_train, label=y_train)
gX_test = xgb.DMatrix(data=X_test, label=y_test)


# In[29]:

ggX_train = xgb.DMatrix(data=a_full, label=dtrain.get_label())
ggX_test = xgb.DMatrix(data=b_full)


# In[30]:

# Test with LR:
def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_
clf = linear_model.Ridge (alpha = 0.6)
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)
y_hat = np.minimum(np.maximum(y_hat, 1.0), 3.0)
fmean_squared_error(y_hat, y_test)


# In[ ]:




# In[ ]:






######

from sklearn.ensemble import ExtraTreesRegressor

X_tra = X_train[:,:-690]
X_tes = X_test[:,:-690]

f = []
for n in [1200]:
    for mf in [345]: #'sqrt', 'log2', 100]:
        for m in [17]:
            etr = ExtraTreesRegressor(n_estimators=n, criterion='mse', max_depth=m, min_samples_split=5,
                                      min_samples_leaf=5, max_features=mf, bootstrap=False,
                                      n_jobs=32, random_state=157, verbose=1)
            etr.fit(X_tra, y_train)
            y_hat = etr.predict(X_tes)
            y_hat = np.minimum(np.maximum(y_hat, 1.0), 3.0)
            ee = fmean_squared_error(y_hat, y_test)
            f.append(ee)
            print n,mf,m
            print ee


a_fu = a_full[:,:-690]
b_fu = b_full[:,:-690]
etr = ExtraTreesRegressor(n_estimators=1200, criterion='mse', max_depth=17, min_samples_split=5,
                          min_samples_leaf=5, max_features=mf, bootstrap=False,
                          n_jobs=32, random_state=151, verbose=1)

etr.fit(a_fu, dtrain.get_label())
y_hat = etr.predict(b_fu)
y_hat = np.minimum(np.maximum(y_hat, 1.0), 3.0)

idx_test = pd.read_pickle(loc % 'LABELS_TEST.df')
idx_test['relevance'] = y_hat
idx_test.to_csv('submission_RenatPawel_combined_features_extratrees_0408_1705.csv')

