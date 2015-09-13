import re
import graphlab as gl
from graphlab.toolkits.feature_engineering import TFIDF

# put in the path to the kaggle data
PATH_TO_JSON = "path/to/data/from/process_html.py"
PATH_TO_TRAIN_LABELS = "path/to/data/train.csv"
PATH_TO_TEST_LABELS = "path/to/data/sampleSubmission.csv"

# a simple method to create some basic features on an SFrame
def create_count_features(sf):
    sf['num_images'] = sf['images'].apply(lambda x: len(x))
    sf['num_links'] = sf['links'].apply(lambda x: len(x))
    sf['num_clean_chars'] = sf['text_clean'].apply(lambda x: len(x))
    return sf

# a simple method to clean the text within an html response
def clean_text(sf):
    sf['text_clean'] = sf['text'].apply(lambda x:
        re.sub(r'[\n\t,.:;()\-\/]+', ' ', ' '.join(x)))
    sf['text_clean'] = sf['text_clean'].apply(lambda x: re.sub(r'\s{2,}', ' ', x))
    sf['text_clean'] = sf['text_clean'].apply(lambda x: x.strip())
    return sf

# a wrapper method around the 2 methods above
def process_dataframe(sf):
    sf = clean_text(sf)
    sf = create_count_features(sf)
    return sf 

# read json blocks from path PATH_TO_JSON
sf = gl.SFrame.read_csv(PATH_TO_JSON, header=False)
sf = sf.unpack('X1',column_name_prefix='')

# read train and test labels from paths PATH_TO_TRAIN_LABELS and PATH_TO_TEST_LABELS
train_labels = gl.SFrame.read_csv(PATH_TO_TRAIN_LABELS)
test_labels = gl.SFrame.read_csv(PATH_TO_TEST_LABELS)

# create a new columns "id" from parsing urlId and drop file columns
train_labels['id'] = train_labels['file'].apply(lambda x: str(x.split('_')[0] ))
train_labels = train_labels.remove_column('file')
test_labels['id'] = test_labels['file'].apply(lambda x: str(x.split('_')[0] ))
test_labels = test_labels.remove_column('file')

# join labels with html data from training and testing SFrames
train = train_labels.join(sf, on='id', how='left')
test = test_labels.join(sf, on='id', how='left')

# call wrapper method process_dataframe on train/test
train =  process_dataframe(train)
test = process_dataframe(test)

# create word counts and remove countwords for both train and test
bow_trn = gl.text_analytics.count_words(train['text_clean'])
bow_trn = bow_trn.dict_trim_by_keys(gl.text_analytics.stopwords())

bow_tst = gl.text_analytics.count_words(test['text_clean'])
bow_tst = bow_tst.dict_trim_by_keys(gl.text_analytics.stopwords())

# add the bag of words to both sframes
train['bow'] = bow_trn
test['bow'] = bow_tst

# create a TFIDF Transformer that is fit on your training data and transform both training and testing data
encoder = gl.feature_engineering.create(train, TFIDF('bow', output_column_name='tfidf'))
train = encoder.transform(train)
test = encoder.transform(test)

# train logistic regression on training data with tf-idf as features and predict on testing data
train = train.dropna()
model = gl.logistic_classifier.create(train, target='sponsored', features=['tfidf'], class_weights='auto')

test = test.dropna()
ypred = model.predict(test)

# create submission.csv
submission = gl.SFrame()
submission['sponsored'] = ypred 
submission['file'] = test['id'].apply(lambda x: x + '_raw_html.txt')
submission.save('submission_version_1.csv', format='csv')