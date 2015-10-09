
# coding: utf-8

# In[1]:

import datetime

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import xgboost as xgb


# In[2]:

print("start")
#df_train = pd.read_pickle('/mnt/sframe/df_train')
df_test = pd.read_pickle('/mnt/sframe/df_test')

#df_train['text'].fillna('', inplace=True)
df_test['text'].fillna('', inplace=True)


# In[ ]:

dtrain = xgb.DMatrix('/mnt/data/orig_dtrain.buffer_mindf20')
dtest = xgb.DMatrix('/mnt/data/orig_dtest.buffer_mindf20')
watchlist  = [(dtrain,'train')]
dtrain.num_row(), dtest.num_row()


# ---
# 
# ## Train

# In[1]:

def generate_submission(param, num_round, filename):
    print("starting %s" % (filename))
    a = datetime.datetime.now()
    print(a)

    bst2 = xgb.train(param, dtrain, num_round, watchlist)

    print(datetime.datetime.now())
    b = datetime.datetime.now() - a
    print(b)

    submission = pd.DataFrame({
            'file': df_test['id'].map(lambda s: s + '_raw_html.txt').reset_index(drop=True),
            'sponsored': bst2.predict(dtest)
        })
    submission.to_csv(filename, index=False)


# In[ ]:

param = {'max_depth': 19, 
         'eta': 0.02, 
         'num_parallel_tree': 4, #10,
         #'gamma': 1.0,
         'colsample_bytree': 0.5, #0.8,
         'subsample': 1.0,
         'min_child_weight': 4, #10,
         'silent':1, 
         'objective':'binary:logistic', 
         'eval_metric':'auc',
         'early_stopping_rounds':20}
num_round = 1 #300


# In[4]:

ps_eta = [0.02, 0.02]
ps_par = [4, 4]
ps_col = [0.5, 0.6]
ps_rounds = [2300, 2300]

ps = zip(ps_eta, ps_par, ps_col, ps_rounds)

for p_eta, p_par, p_col, p_rounds in ps:
    pars = param.copy()
    pars['eta'] = p_eta
    pars['num_parallel_tree'] = p_par
    pars['colsample_bytree'] = p_col
    generate_submission(pars, p_rounds, 'sub_Oct09_depth19_eta_' + str(p_eta)+ '_colsam' + str(p_col) + '_minchild4_par' + str(p_par) + '.csv')


# When tried:
# 
# param = {'max_depth': 19, 
#          'eta': 0.02, 
#          'num_parallel_tree': 4, #10,
#          #'gamma': 1.0,
#          'colsample_bytree': 0.5, #0.8,
#          'subsample': 1.0,
#          'min_child_weight': 4, #10,
#          'silent':1, 
#          'objective':'binary:logistic', 
#          'eval_metric':'auc',
#          'early_stopping_rounds':20}
# num_round = 1 #300
# 
# with overrides
# 
# ps_eta = [0.015, 0.02]
# ps_par = [3, 4]
# ps_col = [0.6, 0.5]
# ps_rounds = [2000, 2000]
# 
# 1 -> 6 hours, LB AUC: 0.96059
# 2 -> 6.25 hours, LB AUC: 0.96121
# 
