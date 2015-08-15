from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

from dataset import get_data
from metaregressor import meta_fit

print("Loading datasets")
dat_x_orig, dat_y_orig, lb_x_orig, lb_ind = get_data()

scaler = preprocessing.StandardScaler().fit(dat_x_orig)
dat_x = scaler.transform(dat_x_orig)
lb_x = scaler.transform(lb_x_orig)
dat_y = dat_y_orig ** 0.75


train_index, cv_index = train_test_split(range(dat_x.shape[0]), test_size=0.1, random_state=102)

subm_y = meta_fit(dat_x, dat_y, train_index, cv_index, lb_x)


