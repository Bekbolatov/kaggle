"""

Beating the Benchmark
Truly Native?
__author__ : David Shinn

"""
from __future__ import print_function

from collections import Counter
import glob
import multiprocessing
import os
import re
import sys
import time

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

print('--- Read training labels')
train_labels = pd.read_csv('data/train_v2.csv')
train_keys = dict([a[1] for a in train_labels.iterrows()])
test_files = set(pd.read_csv('data/sampleSubmission_v2.csv').file.values)

def create_data(filepath):
    values = {}
    filename = os.path.basename(filepath)
    with open(filepath, 'rb') as infile:
        text = infile.read()
    values['file'] = filename
    if filename in train_keys:
        values['sponsored'] = train_keys[filename]
    values['lines'] = text.count('\n')
    values['spaces'] = text.count(' ')
    values['tabs'] = text.count('\t')
    values['braces'] = text.count('{')
    values['brackets'] = text.count('[')
    values['words'] = len(re.split('\s+', text))
    values['length'] = len(text)
    return values

filepaths = glob.glob('data/*/*.txt')
num_tasks = len(filepaths)

p = multiprocessing.Pool()
results = p.imap(create_data, filepaths)
while (True):
    completed = results._index
    print("\r--- Completed {:,} out of {:,}".format(completed, num_tasks), end='')
    sys.stdout.flush()
    time.sleep(1)
    if (completed == num_tasks): break
p.close()
p.join()
df_full = pd.DataFrame(list(results))
print()

print('--- Training random forest')
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
train = df_full[df_full.sponsored.notnull()].fillna(0)
test = df_full[df_full.sponsored.isnull() & df_full.file.isin(test_files)].fillna(0)
clf.fit(train.drop(['file', 'sponsored'], 1), train.sponsored)

print('--- Create predictions and submission')
submission = test[['file']].reset_index(drop=True)
submission['sponsored'] = clf.predict_proba(test.drop(['file', 'sponsored'], 1))[:, 1]
submission.to_csv('native_btb_basic_submission.csv', index=False)

