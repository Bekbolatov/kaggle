import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import numpy as np
import pylab as pl

from gini import normalized_gini
from dataset import get_data

data_root = '/Users/rbekbolatov/repos/gh/bekbolatov/kaggle/events/liberty/cxxnet_proj'
dat_x, dat_y, lb_x, lb_ind = get_data()

def labels(n):
    if n < 8:
        return n - 1
    elif n < 11:
        return 7
    elif n < 25:
        return 8
    else:
        return 9

def labels_1_2(n):
    if n < 11:
        return 0
    else:
        return 1

nplabels = np.vectorize(labels_1_2)

# Scale inputs
scaler = preprocessing.StandardScaler().fit(dat_x)
dat_x = pd.DataFrame(scaler.transform(dat_x))
lb_x = pd.DataFrame(scaler.transform(lb_x))
dat_y = nplabels(dat_y)

# special subset of data
dat_x['label'] = dat_y
#dat_x = dat_x[dat_x['label'] < 6]
dat_y = np.asarray(dat_x['label'])
dat_x = dat_x.drop('label', axis=1)

# split data: train/cv
train_index, test_index = train_test_split(range(dat_x.shape[0]), test_size=0.1, random_state=103)

train_x = dat_x.loc[train_index, :]
train_y = dat_y[train_index]
cv_x = dat_x.loc[test_index, :]
cv_y = dat_y[test_index]


# export for cxxnet
train_x['label'] = train_y
cv_x['label'] = cv_y

cols = train_x.columns.tolist()

train_x = train_x[cols[-1:] + cols[:-1]]
cv_x = cv_x[cols[-1:] + cols[:-1]]

train_x.to_csv(data_root + '/data/train.csv', index=False, header=False)
cv_x.to_csv(data_root + '/data/cv.csv', index=False, header=False)

train_x = train_x.drop('label', axis=1)
cv_x = cv_x.drop('label', axis=1)



#  cxxnet run
#  ...
#  cxxnet done



cxxnet_preds = pd.read_csv(data_root + '/out/pred.csv', header=None)
print(max(cxxnet_preds))

cxxnet_preds[0] == cv_y
cv_error = normalized_gini(cv_y, np.asarray(cxxnet_preds[0], dtype='float32'))
print("Validation Sample Score: {:.10f} (normalized gini).".format(cv_error))
print("MSE = %0.3f" % (sum(np.power( np.asarray(cxxnet_preds[0]) - cv_y, 2 ))/cv_y.shape[0]))

# evaluate

cv_error = normalized_gini(cv_y, cv_y_preds)
print("Validation Sample Score: {:.10f} (normalized gini).".format(cv_error))

preds = pd.DataFrame({"actual": cv_y, "pred": cv_y_preds})
preds.boxplot('pred', 'actual')


# show
max_cy = max(cv_y) * 1.1
pl.scatter(cv_y, cv_y_preds, s=1)
pl.xlim(0, 70)
pl.show()





# analyze
cv_all = np.hstack( (np.vstack((cv_y, cv_y_preds)).T, cv_x))
dat_all = np.hstack( ( dat_y.reshape(-1, 1), dat_x ))
dat_all_df = pd.DataFrame(dat_all)


