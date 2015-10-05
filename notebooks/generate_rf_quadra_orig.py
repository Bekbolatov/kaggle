
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import re
import graphlab as gl
from graphlab.toolkits.feature_engineering import TFIDF, FeatureHasher, QuadraticFeatures

gl.canvas.set_target('ipynb')

PATH_TO_JSON2 = "/mnt/sframe/docs_prod_02/"
PATH_TO_JSON = "/mnt/sframe/docs_prod_05/"
PATH_TO_JSON6 = "/mnt/sframe/docs_prod_06/"
PATH_TO_TRAIN_LABELS = "input/train.csv"
PATH_TO_TEST_LABELS = "input/sampleSubmission.csv"


# ### Read processed documents

# In[2]:

gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 128)
gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY', 100*1024*1024*1024) # 100GB
gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE', 100*1024*1024*1024) # 100GB


# In[3]:

def transf(x):
    return 50.0 * np.log1p(np.log1p(x))


# In[4]:

# documents
sf = gl.SFrame.read_csv(PATH_TO_JSON, header=False, verbose=False)
sf = sf.unpack('X1',column_name_prefix='')
sf['id'] = sf['id'].apply(lambda x: str(x.split('_')[0] ))
sf['num_words'] = sf['text'].apply(lambda xs: transf(len(xs)))


# In[5]:

sf2 = gl.SFrame.read_csv(PATH_TO_JSON2, header=False, verbose=False)
sf2 = sf2.unpack('X1',column_name_prefix='')
sf2['id'] = sf2['id'].apply(lambda x: str(x.split('_')[0] ))


# In[6]:

sf6 = gl.SFrame.read_csv(PATH_TO_JSON6, header=False, verbose=False)
sf6 = sf6.unpack('X1',column_name_prefix='')
sf6['id'] = sf6['id'].apply(lambda x: str(x.split('_')[0] ))
sf6['word2vec'] = sf6['word2vec'].apply(lambda xs: np.array(xs ,dtype='float32').tolist())


# In[7]:

sf_cnt = gl.SFrame()
sf_cnt['id'] = sf2['id']

sf_cnt['a_href'] = sf2['ahref'].apply(lambda x: transf(len(x)))
sf_cnt['par'] = sf2['par'].apply(lambda x: transf(len(x)))
sf_cnt['title'] = sf2['title'].apply(lambda x: transf(len(x)))

sf_cnt['img'] = sf2['img_cnt'].apply(transf)
sf_cnt['btn'] = sf2['misc_button'].apply(transf)
sf_cnt['input'] = sf2['misc_input'].apply(transf)
sf_cnt['li'] = sf2['misc_li'].apply(transf)
sf_cnt['link'] = sf2['misc_link'].apply(transf)
sf_cnt['meta'] = sf2['misc_meta'].apply(transf)

sf_cnt['script_avg'] = sf2['script_avg'].apply(transf)
sf_cnt['script_b_avg'] = sf2['script_b_avg'].apply(transf)
sf_cnt['script_cnt'] = sf2['script_cnt'].apply(transf)
sf_cnt['script_b_cnt'] = sf2['script_b_cnt'].apply(transf)

sf_cnt['style_avg'] = sf2['style_avg'].apply(transf)
sf_cnt['style_cnt'] = sf2['style_cnt'].apply(transf)


# ### Read train/test labels and merge into documents

# In[8]:

# train/test labels
train_labels = gl.SFrame.read_csv(PATH_TO_TRAIN_LABELS, verbose=False)
test_labels = gl.SFrame.read_csv(PATH_TO_TEST_LABELS, verbose=False)
train_labels['id'] = train_labels['file'].apply(lambda x: str(x.split('_')[0] ))
train_labels = train_labels.remove_column('file')
test_labels['id'] = test_labels['file'].apply(lambda x: str(x.split('_')[0] ))
test_labels = test_labels.remove_column('file')


# In[24]:

# join
train = train_labels.join(sf, on='id', how='left')
test = test_labels.join(sf, on='id', how='left')


# In[25]:

train = train.join(sf_cnt, on='id', how='left')
test = test.join(sf_cnt, on='id', how='left')


# In[26]:

train = train.join(sf6, on='id', how='left')
test = test.join(sf6, on='id', how='left')


# In[27]:

features = [
            'a_href',
            'par',
            'title',
            'img',
            'btn',
            'input',
            'li',
            'link',
            'meta',
            'script_avg',
            'script_b_avg',
            'script_cnt',
            'script_b_cnt',
            'style_avg',
            'style_cnt',
            'num_words'
           ]


# In[28]:

for f in features:
    train = train.fillna(f, 0.0)     
    test = test.fillna(f, 0.0)


# In[29]:

train = train.fillna('shinn', {})     
test = test.fillna('shinn', {})

train['shinn'] = train['shinn'].apply(lambda ws: ws if ws else {})
test['shinn'] = test['shinn'].apply(lambda ws: ws if ws else {})

features = features + ['shinn']


# In[30]:

train = train.fillna('word2vec', np.zeros(300))     
test = test.fillna('word2vec', np.zeros(300))

train['word2vec'] = train['word2vec'].apply(lambda ws: ws if ws else np.zeros(300))
test['word2vec'] = test['word2vec'].apply(lambda ws: ws if ws else np.zeros(300))


# In[31]:

train = train.fillna('words', [])     
test = test.fillna('words', [])   

train['words'] = train['words'].apply(lambda ws: ws if ws else [])
test['words'] = test['words'].apply(lambda ws: ws if ws else [])


# In[32]:

train['word_set_size'] = train['words'].apply(lambda ws: len(set(ws)))
test['word_set_size'] = test['words'].apply(lambda ws: len(set(ws)))

train['word_set_size_ratio'] = train.apply(lambda r: r['word_set_size'] * 1.0 / len(r['words']) if len(r['words']) > 0 else 0.0)
test['word_set_size_ratio'] = test.apply(lambda r: r['word_set_size'] * 1.0 / len(r['words']) if len(r['words']) > 0 else 0.0)


# In[33]:

train['text_words'] = train['words'].apply(lambda ws: ' '.join(ws))
test['text_words'] = test['words'].apply(lambda ws: ' '.join(ws))


# In[34]:

features = features + ['word_set_size', 'word_set_size_ratio']


# In[35]:

def text2wordsetsize(text):
    if not text:
        return 0.0
    else:
        return len(set(text.split()))
    
def safe_divide(a, b):
    if b > 0:
        return a * 1.0 / b
    else:
        return 0.0

train['word_set_size2'] = train['text'].apply(text2wordsetsize)
test['word_set_size2'] = test['text'].apply(text2wordsetsize)

train['word_set_size2_ratio'] = train.apply(lambda r: safe_divide(r['word_set_size2'], r['word_set_size']))
test['word_set_size2_ratio'] = test.apply(lambda r: safe_divide(r['word_set_size2'], r['word_set_size']))

features = features + ['word_set_size2', 'word_set_size_ratio2']


# In[ ]:

train = train.unpack('shinn')
test = test.unpack('shinn')


# In[ ]:

def generate_inverse(dataset, feats):
    new_feats = []
    for f in feats:
        new_feat = 'inv_' + f
        new_feats.append(new_feat)
        dataset[new_feat] = dataset.apply(lambda r: safe_divide(1, r[f]))
    return new_feats  

def for_inverse(col):
    return col not in set(['sponsored', 'id', 'text', 'text_words', 'words', 'word2vec', 'bow', 'tfidf', 'shinn'])


cols = [x for x in train.column_names() if for_inverse(x)]
generate_inverse(train, cols)
new_cols = generate_inverse(test, cols)



quadratic = gl.feature_engineering.QuadraticFeatures(features=cols + new_cols)

fit_quadratic = quadratic.fit(train)
train = fit_quadratic.transform(train)
test = fit_quadratic.transform(test)


train.remove_columns([col for col in train.column_names() if col.startswith('inv_')])
test.remove_columns([col for col in test.column_names() if col.startswith('inv_')])


# ### Generate BOW

# In[ ]:

bow_trn = gl.text_analytics.count_words(train['text'])
bow_trn = bow_trn.dict_trim_by_keys(gl.text_analytics.stopwords())

bow_tst = gl.text_analytics.count_words(test['text'])
bow_tst = bow_tst.dict_trim_by_keys(gl.text_analytics.stopwords())

train['bow'] = bow_trn
test['bow'] = bow_tst


# ### Generate TF-IDF

# In[ ]:

encoder = gl.feature_engineering.create(train, TFIDF('bow', output_column_name='tfidf', min_document_frequency=5e-5))
train = encoder.transform(train)
test = encoder.transform(test)


# In[ ]:

train['tfidf'] = train['tfidf'].fillna({})
test['tfidf'] = test['tfidf'].fillna({})

train['tfidf'] = train['tfidf'].apply(lambda x: x if x else {})
test['tfidf'] = test['tfidf'].apply(lambda x: x if x else {})


# ## Features

# In[ ]:

def feature_col(col):
    return col not in set(['sponsored', 'id', 'text', 'text_words', 'word2vec', 'words', 'bow'])

features = [f for f in train.column_names() if feature_col(f)]


# ### Split training set for cross-validation

# In[ ]:

train_train, train_cv = train.random_split(0.80, seed=107)


# In[ ]:

train_train_train, train_train_cv = train_train.random_split(0.75, seed=7)


# In[ ]:

TRAIN, CV = train_train_cv.random_split(0.50, seed=113)


# In[ ]:

TRAIN.shape, CV.shape


# # Generate Submission Model

# Started at 20:36:00am

# In[ ]:

features


# In[ ]:

get_ipython().magic(u'pinfo gl.classifier.random_forest_classifier.create')


# In[ ]:

model200=model


# In[ ]:

train.head()


# In[ ]:

feature_imp = model200.get_feature_importance()


# In[ ]:

gl.canvas.set_target('ipynb')


# In[ ]:

fi = list(gl.load_sframe('feature_importance.csv')['feature'])


# In[ ]:

def decode_dict(a):
    #a = "shinn[\"hello\"]"
    dic, word = a.split('[')
    _, word, _ = word.split('\"')
    return dic, word

def get_or_else(dic, word, no=0.0):
    if dic.has_key(word) and dic[word]:
        return dic[word]
    else:
        return no
    
def value_it(a, data, out):
    if '[' in a:
        dic, word = decode_dict(a)
        out[dic + '.' + word] = data[dic].apply(lambda d: get_or_else(d, word))
    else:
        out[a] = data[a]


# In[ ]:

CV_l = gl.SFrame()
for a in fi[0:150]:
    value_it(a, CV, CV_l)
    
TRAIN_l = gl.SFrame()
for a in fi[0:150]:
    value_it(a, TRAIN, TRAIN_l)    
    
CV_l['sponsored'] = CV['sponsored']
TRAIN_l['sponsored'] = TRAIN['sponsored']
new_feats = set(CV_l.column_names()).difference(set(['sponsored']))


# In[ ]:

'sponsored' in new_feats


# In[ ]:

new_feats


# In[ ]:

a = {}
get_or_else(a, 's')


# In[ ]:

TRAIN_l = TRAIN.unpack('shinn')
CV_l = CV.unpack('shinn')


# In[ ]:

CV_l = CV_l.unpack('tfidf5e5')


# In[ ]:

'sponsored' in set(CV_l.column_names())


# In[ ]:

CV_l['tfidf5e5.copyright'].show()


# In[ ]:

3 + 4


# In[ ]:

CV[CV['sponsored'] == 0]['a_href'].sum()  / CV[CV['sponsored'] == 0].shape[0]


# In[ ]:

model = gl.classifier.boosted_trees_classifier.create(train, target='sponsored',
                                                      #features=features + ['tfidf_hashed_18'],
                                                      features=features + ['tfidf5e5'],
                                                      max_depth=6,
                                                      step_size=0.2,
                                                      max_iterations=300,
                                                      column_subsample=0.3,
                                                      row_subsample=1.0,
                                                      class_weights='auto')


# In[ ]:

model = gl.classifier.random_forest_classifier.create(train, target='sponsored',
                                                      features=features, # + ['tfidf', 'word2vec'],
                                                      num_trees=200,
                                                      max_depth=150,
                                                      validation_set=None,
                                                      column_subsample=0.25,
                                                      row_subsample=1.0,
                                                      class_weights='auto')


# In[ ]:

model = gl.classifier.boosted_trees_classifier.create(train, target='sponsored',
                                                      #features=features + ['tfidf_hashed_18'],
                                                      features=features + ['tfidf'],
                                                      max_depth=6,
                                                      step_size=0.2,
                                                      max_iterations=300,
                                                      column_subsample=0.3,
                                                      row_subsample=1.0,
                                                      class_weights='auto',
                                                      validation_set=None)


# In[ ]:

lr_model = gl.logistic_classifier.create(TRAIN_l, target='sponsored', 
                                      features=new_feats,
                                      validation_set=CV_l,
                                      class_weights='auto',
                                      max_iterations=10,
                                      l2_penalty=0.00,
                                      l1_penalty=0.00)


# In[ ]:

svm_model = gl.svm_classifier.create(train, target='sponsored', 
                                      features=['tfidf_hashed'],
                                      validation_set=None,                                           
                                      class_weights='auto',
                                      max_iterations=20)


# ### Output model

# In[ ]:

import datetime
print(datetime.datetime.now())


# In[ ]:

ypred = model.predict(test, 'probability')

submission = gl.SFrame()
submission['file'] = test['id'].apply(lambda x: x + '_raw_html.txt')
submission['sponsored'] = ypred 
#submission.save('submission_version_4.csv', format='csv')

submission = submission.to_dataframe()
submission.to_csv('submission_rf_quadras.csv', index=False, float_format='%1.8f')


# In[ ]:

model.save('/mnt/sframe/model_RF_200_150_noword2vec')


# In[ ]:

test


# # Experiment

# ### Split train into *train_train*/*train_cv*

# In[ ]:

model = gl.logistic_classifier.create(train_train, target='sponsored', 
                                      features=features + ['tfidf'],
                                      validation_set=train_cv,
                                      class_weights='auto',
                                      max_iterations=30,
                                      feature_rescaling=True,
                                      l2_penalty=0.00,
                                      l1_penalty=0.00)


# In[ ]:

results = gl.SFrame()
results['id'] = train_cv['id']
results['actual'] = train_cv['sponsored']
results['predicted'] = model.predict(train_cv)


# In[ ]:

train_cv.unpack('tfidf')


# In[ ]:

FN.shape, FP.shape


# In[ ]:

FN = results[results['actual'] > results['predicted']]
FP = results[results['actual'] < results['predicted']]


# In[ ]:

FN[720:730]


# In[ ]:

FP


# In[ ]:

model.evaluate(train_cv)


# In[ ]:

results = model.evaluate(train_cv, metric='roc_curve')
a = results['roc_curve']

fpr = list(a['fpr'])
tpr = list(a['tpr'])
fpr[0] = 1.0
tpr[0] = 1.0
fpr = np.array(fpr)
tpr = np.array(tpr)

AUC = np.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + (tpr[:-1] - tpr[1:])/2))
plt.plot(fpr, tpr)
print('AUC = %f'%AUC)


# In[ ]:

train_cv.remove_column('tfidf_hashed17')


# In[ ]:

svm_model = gl.svm_classifier.create(train_train, target='sponsored', 
                                      features=features + ['tfidf_hashed'],
                                      validation_set=train_cv,                                           
                                      class_weights='auto',
                                      max_iterations=15)


# In[ ]:

svm_model


# In[ ]:

train_cv['margin'] = svm_model.predict(train_cv, output_type='margin')
preds = train_cv[['sponsored', 'margin']].sort('margin')
train_cv.remove_column('margin')

pd_preds = preds.to_dataframe()
pd_preds['number'] = 1.0

pd_preds_cum = pd_preds.cumsum()

total_positives = np.asarray(pd_preds_cum['sponsored'])[-1]
total = np.asarray(pd_preds_cum['number'])[-1]
total_negatives = total - total_positives

pd_preds_cum['FN'] = pd_preds_cum['sponsored']
pd_preds_cum['TN'] = pd_preds_cum['number'] - pd_preds_cum['sponsored']

pd_preds_cum['TP'] = total_positives - pd_preds_cum['FN']
pd_preds_cum['FP'] = total - total_positives - pd_preds_cum['TN']

pd_preds_cum['fpr'] = pd_preds_cum['FP'] / (pd_preds_cum['FP'] + pd_preds_cum['TN'])
pd_preds_cum['tpr'] = pd_preds_cum['TP'] / (pd_preds_cum['TP'] + pd_preds_cum['FN'])



# In[ ]:

a = pd_preds_cum

fpr = list(a['fpr'])
tpr = list(a['tpr'])
fpr[0] = 1.0
tpr[0] = 1.0
fpr = np.array(fpr)
tpr = np.array(tpr)

AUC = np.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + (tpr[:-1] - tpr[1:])/2))
plt.plot(fpr, tpr)
print('AUC = %f'%AUC)


# In[ ]:

train_cv['margin'] = svm_model.predict(train_cv, output_type='margin')
preds = train_cv[['sponsored', 'margin']]
preds['margin'].show()


# In[ ]:

(preds[preds['margin'] < 55]['sponsored']).sum()


# In[ ]:

ts = np.arange(-22, 50, 0.1)
[for t in ts]


# In[ ]:

svm_model.evaluate(train_cv)


# In[ ]:

results = svm_model.evaluate(train_cv, metric='roc_curve')
a = results['roc_curve']

fpr = list(a['fpr'])
tpr = list(a['tpr'])
fpr[0] = 1.0
tpr[0] = 1.0
fpr = np.array(fpr)
tpr = np.array(tpr)

AUC = np.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + (tpr[:-1] - tpr[1:])/2))
plt.plot(fpr, tpr)
print('AUC = %f'%AUC)


# In[ ]:

results


# # Save Datasets?

# In[ ]:

train.save('/mnt/sframe/quadra_train')
test.save('/mnt/sframe/quadra_test')


# In[ ]:

train_train = gl.load_sframe('/mnt/sframe/shinn_split_train_train')
train_cv = gl.load_sframe('/mnt/sframe/shinn_split_train_cv')


# In[ ]:

train_train.head()


# # Junk

# In[ ]:

model_17 = gl.logistic_classifier.create(train_train, target='sponsored', 
                                      features=['tfidf_hashed_17'],
                                      validation_set=train_cv,
                                      class_weights=None, #'auto',
                                      max_iterations=7,
                                      feature_rescaling=True,
                                      l2_penalty=0.00,
                                      l1_penalty=0.00)


# In[ ]:

model_17.evaluate(train_cv)


# In[ ]:

results = model_17.evaluate(train_cv, metric='roc_curve')
a = results['roc_curve']

fpr = list(a['fpr'])
tpr = list(a['tpr'])
fpr[0] = 1.0
tpr[0] = 1.0
fpr = np.array(fpr)
tpr = np.array(tpr)

AUC = np.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + (tpr[:-1] - tpr[1:])/2))
plt.plot(fpr, tpr)
print('AUC = %f'%AUC)


# In[ ]:

gl.svm_classifier.create(train_train, target='sponsored', 
                                      features=['tfidf_hashed_15'],
                                      validation_set=train_cv,                                           
                                      class_weights='auto',
                                      max_iterations=40)


# # Continue with data

# In[ ]:

train = gl.load_sframe('/mnt/sframe/counts_and_tfidf_hashed_18_train')
test = gl.load_sframe('/mnt/sframe/counts_and_tfidf_hashed_18_test')

train_train = gl.load_sframe('/mnt/sframe/num_words_counts_and_tfidf_hashed_18_split_train_train')
train_cv = gl.load_sframe('/mnt/sframe/num_words_counts_and_tfidf_hashed_18_split_train_cv')

TRAIN, CV = train_cv.random_split(0.50, seed=113)


# In[ ]:

train.save('/mnt/sframe/shinn_train')
test.save('/mnt/sframe/shinn_test')

train_train.save('/mnt/sframe/shinn_split_train_train')
train_cv.save('/mnt/sframe/shinn_split_train_cv')
#train_train = gl.load_sframe('/mnt/sframe/tfidf_hashed_16_split_train_train')
#train_cv = gl.load_sframe('/mnt/sframe/tfidf_hashed_16_split_train_cv')


# ## Try classifiers

# In[ ]:

def text2wordsetsize(text):
    if not text:
        return 0.0
    else:
        return len(set(text.split()))
    
def safe_divide(a, b):
    if b > 0:
        return a * 1.0 / b
    else:
        return 0.0

TRAIN['word_set_size2'] = TRAIN['text'].apply(text2wordsetsize)
CV['word_set_size2'] = CV['text'].apply(text2wordsetsize)

TRAIN['word_set_size2_ratio'] = TRAIN.apply(lambda r: safe_divide(r['word_set_size2'], r['word_set_size']))
CV['word_set_size2_ratio'] = CV.apply(lambda r: safe_divide(r['word_set_size2'], r['word_set_size']))


# In[ ]:

TRAIN.head()


# In[ ]:

model_boosted.get_feature_importanceature_importance()


# In[ ]:

# model_rf = gl.classifier.random_forest_classifier.create(train_train, target='sponsored',
#                                                       features=features + ['word2vec'],
#                                                       num_trees=10,
#                                                       max_depth=200,
#                                                       column_subsample=0.15,
#                                                       row_subsample=1.0,
#                                                       class_weights='auto',
#                                                       validation_set=train_cv)
model_boosted = gl.classifier.boosted_trees_classifier.create(TRAIN, target='sponsored',
                                                      features=features + ['shinn', # 'word_set_size2', 
                                                                           #'word_set_size2_ratio',
                                                                           'tfidf'], #, 'word2vec'],
                                                      max_depth=2,
                                                      step_size=0.15,  #0.2
                                                      max_iterations=15,
                                                      column_subsample=0.4,
                                                      row_subsample=1.0,
                                                      class_weights='auto',
                                                      validation_set=CV)


# In[ ]:

results = model_boosted.evaluate(CV, metric='roc_curve')
a = results['roc_curve']

fpr = list(a['fpr'])
tpr = list(a['tpr'])
fpr[0] = 1.0
tpr[0] = 1.0
fpr = np.array(fpr)
tpr = np.array(tpr)

AUC = np.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + (tpr[:-1] - tpr[1:])/2))
plt.plot(fpr, tpr)
print('AUC = %f'%AUC)


# In[ ]:

model_rf.get_feature_importance().print_rows(60)


# In[ ]:

model_rf = gl.classifier.random_forest_classifier.create(TRAIN, target='sponsored',
                                                      features=features + ['shinn', 
                                                                           'tfidf',
                                                                           'quadratic_features'
                                                                          ], #, 'word2vec'],
                                                      num_trees=10, #100,
                                                      max_depth=150,
                                                      column_subsample=0.45,
                                                      row_subsample=1.0,
                                                      class_weights='auto',
                                                      validation_set=CV)


# In[ ]:

results = model_rf.evaluate(train_cv, metric='roc_curve')
a = results['roc_curve']

fpr = list(a['fpr'])
tpr = list(a['tpr'])
fpr[0] = 1.0
tpr[0] = 1.0
fpr = np.array(fpr)
tpr = np.array(tpr)

AUC = np.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + (tpr[:-1] - tpr[1:])/2))
plt.plot(fpr, tpr)
print('AUC = %f'%AUC)


# In[ ]:

def generate_inverse(dataset, feats):
    new_feats = []
    for f in feats:
        new_feat = 'inv_' + f
        new_feats.append(new_feat)
        dataset[new_feat] = dataset.apply(lambda r: safe_divide(1, r[f]))
    return new_feats            


# In[ ]:

features.remove('shinn')


# In[ ]:

features


# In[ ]:

new_feats = generate_inverse(TRAIN, features)
generate_inverse(CV, features)


# In[ ]:

new_feats + features

quadratic = gl.feature_engineering.QuadraticFeatures(features=features + new_feats)
fit_quadratic = quadratic.fit(TRAIN)
TRAIN = fit_quadratic.transform(TRAIN)
CV = fit_quadratic.transform(CV)


# In[ ]:

a = TRAIN.column_names()


# In[ ]:

def for_inverse(col):
    return col not in set(['sponsored', 'id', 'text', 'text_words', 'words', 'bow', 'tfidf', 'shinn'])
[x for x in .column_names() if for_inverse(x)]


# In[ ]:

get_ipython().magic(u'pinfo TRAIN.remove_columns')


# In[ ]:




# In[ ]:

T_TRAIN = TRAIN['word2vec'].unpack()
T_CV = CV['word2vec'].unpack()

T_TRAIN['sponsored'] = TRAIN['sponsored']
T_CV['sponsored'] = CV['sponsored']


# In[ ]:

model_nn = gl.nearest_neighbor_classifier.create(TRAIN, target='sponsored', 
                                                 features=['tfidf'] # 'word2vec']
                                                )


# In[ ]:

preds = model_nn.predict(CV, max_neighbors=200, output_type='probability')


# In[ ]:

preds


# In[ ]:

T_CV['prob'] = preds
preds = T_CV[['sponsored', 'prob']].sort('prob')
T_CV.remove_column('prob')

pd_preds = preds.to_dataframe()
pd_preds['number'] = 1.0

pd_preds_cum = pd_preds.cumsum()

total_positives = np.asarray(pd_preds_cum['sponsored'])[-1]
total = np.asarray(pd_preds_cum['number'])[-1]
total_negatives = total - total_positives

pd_preds_cum['FN'] = pd_preds_cum['sponsored']
pd_preds_cum['TN'] = pd_preds_cum['number'] - pd_preds_cum['sponsored']

pd_preds_cum['TP'] = total_positives - pd_preds_cum['FN']
pd_preds_cum['FP'] = total - total_positives - pd_preds_cum['TN']

pd_preds_cum['fpr'] = pd_preds_cum['FP'] / (pd_preds_cum['FP'] + pd_preds_cum['TN'])
pd_preds_cum['tpr'] = pd_preds_cum['TP'] / (pd_preds_cum['TP'] + pd_preds_cum['FN'])



# In[ ]:

a = pd_preds_cum

fpr = list(a['fpr'])
tpr = list(a['tpr'])
fpr[0] = 1.0
tpr[0] = 1.0
fpr = np.array(fpr)
tpr = np.array(tpr)

AUC = np.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + (tpr[:-1] - tpr[1:])/2))
plt.plot(fpr, tpr)
print('AUC = %f'%AUC)


# In[ ]:

model1 = gl.classifier.random_forest_classifier.create(train_train, target='sponsored',
                                                      features=features + ['tfidf'], #, 'word2vec'],
                                                      num_trees=90, #100,
                                                      max_depth=150,
                                                      column_subsample=0.45,
                                                      row_subsample=1.0,
                                                      class_weights='auto',
                                                      validation_set=train_cv)


# In[ ]:

results = model1.evaluate(train_cv, metric='roc_curve')
a = results['roc_curve']

fpr = list(a['fpr'])
tpr = list(a['tpr'])
fpr[0] = 1.0
tpr[0] = 1.0
fpr = np.array(fpr)
tpr = np.array(tpr)

AUC = np.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + (tpr[:-1] - tpr[1:])/2))
plt.plot(fpr, tpr)
print('AUC = %f'%AUC)


# In[ ]:

model = gl.classifier.random_forest_classifier.create(train_train, target='sponsored',
                                                      features=features + ['tfidf'], #, 'word2vec'],
                                                      num_trees=10,
                                                      max_depth=150,
                                                      column_subsample=0.45,
                                                      row_subsample=1.0,
                                                      class_weights='auto',
                                                      validation_set=train_cv)


# In[ ]:

results = model.evaluate(train_cv, metric='roc_curve')
a = results['roc_curve']

fpr = list(a['fpr'])
tpr = list(a['tpr'])
fpr[0] = 1.0
tpr[0] = 1.0
fpr = np.array(fpr)
tpr = np.array(tpr)

AUC = np.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + (tpr[:-1] - tpr[1:])/2))
plt.plot(fpr, tpr)
print('AUC = %f'%AUC)


# In[ ]:

model.get_feature_importance().print_rows(num_rows=30, num_columns=2) 


# In[ ]:

train_train['tfidf_nonone'] = train_train['tfidf'].apply(lambda x: x if x else {})
train_cv['tfidf_nonone'] = train_cv['tfidf'].apply(lambda x: x if x else {})


# In[ ]:

train_train['tfidf_nonone'] = train_train['tfidf_nonone'].fillna({})
train_cv['tfidf_nonone'] = train_cv['tfidf_nonone'].fillna({})


# In[ ]:

svm_model = gl.svm_classifier.create(train_train, target='sponsored', 
                                      features=features + ['tfidf_nonone'], #features + ['tfidf'],
                                      validation_set=train_cv,                                           
                                      class_weights='auto',
                                      max_iterations=40)


# In[ ]:

train_cv['margin'] = svm_model.predict(train_cv, output_type='margin')
preds = train_cv[['sponsored', 'margin']].sort('margin')
train_cv.remove_column('margin')

pd_preds = preds.to_dataframe()
pd_preds['number'] = 1.0

pd_preds_cum = pd_preds.cumsum()

total_positives = np.asarray(pd_preds_cum['sponsored'])[-1]
total = np.asarray(pd_preds_cum['number'])[-1]
total_negatives = total - total_positives

pd_preds_cum['FN'] = pd_preds_cum['sponsored']
pd_preds_cum['TN'] = pd_preds_cum['number'] - pd_preds_cum['sponsored']

pd_preds_cum['TP'] = total_positives - pd_preds_cum['FN']
pd_preds_cum['FP'] = total - total_positives - pd_preds_cum['TN']

pd_preds_cum['fpr'] = pd_preds_cum['FP'] / (pd_preds_cum['FP'] + pd_preds_cum['TN'])
pd_preds_cum['tpr'] = pd_preds_cum['TP'] / (pd_preds_cum['TP'] + pd_preds_cum['FN'])

#   show
a = pd_preds_cum

fpr = list(a['fpr'])
tpr = list(a['tpr'])
fpr[0] = 1.0
tpr[0] = 1.0
fpr = np.array(fpr)
tpr = np.array(tpr)

AUC = np.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + (tpr[:-1] - tpr[1:])/2))
plt.plot(fpr, tpr)
print('AUC = %f'%AUC)


# In[ ]:

model_boosted = gl.classifier.boosted_trees_classifier.create(train_train, target='sponsored',
                                                      features=features + ['tfidf_hashed_18'],
                                                      max_depth=6,
                                                      step_size=0.2,
                                                      max_iterations=300,
                                                      column_subsample=0.3,
                                                      row_subsample=1.0,
                                                      class_weights='auto',
                                                      validation_set=train_cv)


# In[ ]:

results = model_boosted.evaluate(train_cv, metric='roc_curve')
a = results['roc_curve']

fpr = list(a['fpr'])
tpr = list(a['tpr'])
fpr[0] = 1.0
tpr[0] = 1.0
fpr = np.array(fpr)
tpr = np.array(tpr)

AUC = np.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + (tpr[:-1] - tpr[1:])/2))
plt.plot(fpr, tpr)
print('AUC = %f'%AUC)


# In[ ]:

gl.boosted_trees_classifier.get_default_options()


# In[ ]:

TRAIN, CV = train_cv.random_split(0.50, seed=113)


# In[ ]:

CV.shape


# In[ ]:

model_boosted = gl.classifier.boosted_trees_classifier.create(TRAIN, target='sponsored',
                                                      features=features + ['tfidf_hashed_18'],
                                                      max_depth=6,
                                                      step_size=0.2,
                                                      max_iterations=500,
                                                      column_subsample=0.3,
                                                      row_subsample=1.0,
                                                      class_weights='auto',
                                                      validation_set=CV)


# In[ ]:

results = model_boosted.evaluate(CV, metric='roc_curve')
a = results['roc_curve']

fpr = list(a['fpr'])
tpr = list(a['tpr'])
fpr[0] = 1.0
tpr[0] = 1.0
fpr = np.array(fpr)
tpr = np.array(tpr)

AUC = np.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + (tpr[:-1] - tpr[1:])/2))
plt.plot(fpr, tpr)
print('AUC = %f'%AUC)


# In[ ]:

model_boosted = gl.classifier.boosted_trees_classifier.create(train_train, target='sponsored',
                                                      features=features + ['tfidf_hashed_18'],
                                                      max_depth=6,
                                                      step_size=0.2,
                                                      max_iterations=400,
                                                      column_subsample=0.4,
                                                      row_subsample=1.0,
                                                      class_weights='auto',
                                                      validation_set=train_cv)


# In[ ]:

train_cv_pred = gl.SFrame()
train_cv_pred['pred'] =model_boosted.predict(train_cv)
train_cv_pred['actual'] = train_cv['sponsored']
train_cv_pred['id'] = train_cv['id']


# In[ ]:

pred_over = train_cv_pred[train_cv_pred['pred'] > train_cv_pred['actual']]
pred_under = train_cv_pred[train_cv_pred['pred'] < train_cv_pred['actual']]


# In[ ]:

for z in list(pred_under.sample(0.02)['id'].apply(lambda x: 'aws s3 cp s3://sparkydotsdata/kaggle/native/orig/' + x + '_raw_html.txt ' + x +'raw_html')):
    print(z) 


# In[ ]:

results = model_boosted.evaluate(train_cv, metric='roc_curve')
a = results['roc_curve']

fpr = list(a['fpr'])
tpr = list(a['tpr'])
fpr[0] = 1.0
tpr[0] = 1.0
fpr = np.array(fpr)
tpr = np.array(tpr)

AUC = np.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + (tpr[:-1] - tpr[1:])/2))
plt.plot(fpr, tpr)
print('AUC = %f'%AUC)


# In[ ]:

def custom_evaluator(model, train, test):
    results = model.evaluate(test, metric='roc_curve')
    a = results['roc_curve']

    fpr = list(a['fpr'])
    tpr = list(a['tpr'])
    fpr[0] = 1.0
    tpr[0] = 1.0
    fpr = np.array(fpr)
    tpr = np.array(tpr)

    AUC = np.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + (tpr[:-1] - tpr[1:])/2))
    return {'AUC': AUC}


# In[ ]:

job0 = job


# In[ ]:

params = dict([
        ('target', 'sponsored'),
        ('features', [features + ['tfidf_hashed_18']]),
        ('max_depth', [6]),
        ('step_size', [0.2]),
        ('max_iterations', [100, 150, 200]),
        ('column_subsample', [0.4]),
        ('validation_set', [None])
    ])

job = gl.grid_search.create((TRAIN, CV), 
                              gl.boosted_trees_classifier.create, 
                              params, 
                              evaluator=custom_evaluator)
job.get_results()


# In[ ]:

models = job.get_models()


# In[ ]:

results = job.get_results()


# In[ ]:

results = results.to_dataframe()


# In[ ]:

results.boxplot('AUC', by='max_iterations')


# In[ ]:




# In[ ]:

results.sort('AUC', ascending=False)


# In[ ]:

len(models)


# In[ ]:

job.get_metrics()


# In[ ]:

aa = scipy.stats.distributions.expon(.1)


# In[ ]:

from sklearn.ensemble import RandomForestClassifier
import pandas as pd


# In[ ]:

clf = RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=0)
#train = df_full[df_full.sponsored.notnull()].fillna(0)
#test = df_full[df_full.sponsored.isnull() & df_full.file.isin(test_files)].fillna(0)
# clf.fit(train.drop(['file', 'sponsored'], 1), train.sponsored)

# print('--- Create predictions and submission')
# submission = test[['file']].reset_index(drop=True)
# submission['sponsored'] = clf.predict_proba(test.drop(['file', 'sponsored'], 1))[:, 1]
# submission.to_csv('native_btb_basic_submission.csv', index=False)


# In[ ]:

shiTRAIN = shiTRAIN.to_dataframe()
shiCV = shiCV.to_dataframe()


# In[ ]:

for col in shiTRAIN.column_names:
    shiTRAIN[col] = 


# In[ ]:

clf.fit(shiTRAIN, shiTRAIN_label)


# In[ ]:




# In[ ]:

shiTRAIN = TRAIN.unpack('shinn')
shiCV = CV.unpack('shinn')


# In[ ]:

shiTRAIN_label = np.asarray(shiTRAIN['sponsored'])
shiCV_label = np.asarray(shiCV['sponsored'])


# In[ ]:

shiTRAIN_label = np.asarray(shiTRAIN_label, float)
shiCV_label = np.asarray(shiCV_label, float)


# In[ ]:

shiTRAIN_tf = shiTRAIN['tfidf5e5']
shiCV_tf = shiCV['tfidf5e5']


# In[ ]:

shiTRAIN.remove_columns(['tfidf5e5'])
shiCV.remove_columns(['tfidf5e5'])
# shiTRAIN.remove_columns(['text', 'bow', 'sponsored', 'id'])
# shiCV.remove_columns(['text', 'bow', 'sponsored', 'id'])


# In[ ]:

shiTRAIN = shiTRAIN.to_dataframe drop('text', 1)
shiCV = shiCV.drop('text', 1)


# In[ ]:

shiTRAIN.head()

