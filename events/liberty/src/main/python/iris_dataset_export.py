import pandas as pd
from sklearn.cross_validation import train_test_split

data_root = '/Users/rbekbolatov/repos/gh/bekbolatov/kaggle/events/liberty/cxxnet_proj'

from sklearn import datasets

iris = datasets.load_iris()
dat_x = pd.DataFrame(iris.data)
dat_y = iris.target

train_index, test_index = train_test_split(range(dat_x.shape[0]), test_size=0.15, random_state=103)

train_x = dat_x.loc[train_index, :]
train_y = dat_y[train_index]
cv_x = dat_x.loc[test_index, :]
cv_y = dat_y[test_index]

train_x['label'] = train_y
cv_x['label'] = cv_y

cols = train_x.columns.tolist()

train_x = train_x[cols[-1:] + cols[:-1]]
cv_x = cv_x[cols[-1:] + cols[:-1]]

train_x.to_csv(data_root + '/data/train.csv', index=False, header=False)
cv_x.to_csv(data_root + '/data/cv.csv', index=False, header=False)

train_x = train_x.drop('label', axis=1)
cv_x = cv_x.drop('label', axis=1)
