from sklearn import preprocessing
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from dataset import get_data

dat_x, dat_y, lb_x, lb_ind = get_data()

#scaler = preprocessing.StandardScaler().fit(dat_x)
#dat_x = scaler.transform(dat_x)
#lb_x = scaler.transform(lb_x)

model = KNeighborsRegressor(n_neighbors=2, weights='average', p=1)
model.fit(dat_x, dat_y)

lb_pred = model.predict(lb_x)


submission = pd.DataFrame({"Id": lb_ind, "Hazard": lb_pred})
submission = submission.set_index('Id')
submission.to_csv('../subm/KNN_2.csv')
