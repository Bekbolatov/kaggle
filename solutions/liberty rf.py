import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.feature_extraction import DictVectorizer as DV

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

ids = test['Id']
y = train['Hazard']
train = train.drop(['Hazard', 'Id'], axis=1)
test = test.drop(['Id'], axis=1)

# get the categorical columns
fact_cols = ['T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11', 
'T1_V12', 'T1_V15', 'T1_V16', 'T1_V17', 'T2_V3', 'T2_V5', 'T2_V11', 'T2_V12',
'T2_V13']
fact_train = train[fact_cols]
fact_test = test[fact_cols]

#put the numerical as matrix
num_train_data = train.drop(fact_cols, axis=1).as_matrix()
num_test_data = test.drop(fact_cols, axis=1).as_matrix()

#transform the categorical to dict
dict_train_data = fact_train.T.to_dict().values()
dict_test_data = fact_test.T.to_dict().values()

#vectorize
vectorizer = DV(sparse = False)
vec_train_data = vectorizer.fit_transform(dict_train_data)
vec_test_data = vectorizer.fit_transform(dict_test_data)

#merge numerical and categorical sets
x_train = np.hstack((num_train_data, vec_train_data))
x_test = np.hstack((num_test_data, vec_test_data))

print np.shape(x_train)
print np.shape(x_test)

rf = ensemble.RandomForestRegressor(n_estimators=200, max_depth=9)
rf.fit(x_train, y)
pred = rf.predict(x_test)

preds = pd.DataFrame({"Id": ids, "Hazard": pred})
preds = preds[['Id', 'Hazard']]
preds.to_csv('submit.csv', index=False)