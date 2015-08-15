from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing

from gini import normalized_gini
from dataset import get_data

dat_x, dat_y_orig, lb_x, lb_ind = get_data()
dat_y = dat_y_orig ** 0.75


train_index, test_index = train_test_split(range(dat_x.shape[0]), test_size=0.1, random_state=102)
neigh = KNeighborsRegressor(n_neighbors=200, weights='distance', p=1)

train_X_unscaled = dat_x.iloc[train_index, :]
train_y = dat_y[train_index]
test_X_unscaled = dat_x.iloc[test_index, :]
test_y = dat_y[test_index]

scaler = preprocessing.StandardScaler().fit(train_X_unscaled)
train_X = scaler.transform(train_X_unscaled)
test_X = scaler.transform(test_X_unscaled)

neigh.fit(train_X, train_y)
preds = neigh.predict(test_X)
pred_error = normalized_gini(test_y, preds)
print("Pred error: {:.4f}".format(pred_error))
