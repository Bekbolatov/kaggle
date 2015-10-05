import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import re
import graphlab as gl
from graphlab.toolkits.feature_engineering import TFIDF, FeatureHasher, QuadraticFeatures

#gl.canvas.set_target('ipynb')

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

print("continue joins")
# In[25]:

train = train.join(sf_cnt, on='id', how='left')
test = test.join(sf_cnt, on='id', how='left')


# In[26]:

train = train.join(sf6, on='id', how='left')
test = test.join(sf6, on='id', how='left')

print("fill nas")
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

print("unpack shinn")

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

print("generate inverses")
cols = [x for x in train.column_names() if for_inverse(x)]
generate_inverse(train, cols)
new_cols = generate_inverse(test, cols)



quadratic = gl.feature_engineering.QuadraticFeatures(features=cols + new_cols)

print("generated quadra")
fit_quadratic = quadratic.fit(train)
train = fit_quadratic.transform(train)
test = fit_quadratic.transform(test)


train.remove_columns([col for col in train.column_names() if col.startswith('inv_')])
test.remove_columns([col for col in test.column_names() if col.startswith('inv_')])


# ### Generate BOW

# In[ ]:

print("generate bow")

bow_trn = gl.text_analytics.count_words(train['text'])
bow_trn = bow_trn.dict_trim_by_keys(gl.text_analytics.stopwords())

bow_tst = gl.text_analytics.count_words(test['text'])
bow_tst = bow_tst.dict_trim_by_keys(gl.text_analytics.stopwords())

train['bow'] = bow_trn
test['bow'] = bow_tst


# ### Generate TF-IDF

# In[ ]:

print("generate TFIDF")

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

print("run RF classifier")

model = gl.classifier.random_forest_classifier.create(train, target='sponsored',
                                                      features=features, # + ['tfidf', 'word2vec'],
                                                      num_trees=200,
                                                      max_depth=150,
                                                      validation_set=None,
                                                      column_subsample=0.25,
                                                      row_subsample=1.0,
                                                      class_weights='auto')


import datetime
print(datetime.datetime.now())



ypred = model.predict(test, 'probability')

submission = gl.SFrame()
submission['file'] = test['id'].apply(lambda x: x + '_raw_html.txt')
submission['sponsored'] = ypred 

submission = submission.to_dataframe()
submission.to_csv('submission_rf_quadras.csv', index=False, float_format='%1.8f')


train.save('/mnt/sframe/quadra_train')
test.save('/mnt/sframe/quadra_test')

model.save('/mnt/sframe/model_quadras')

