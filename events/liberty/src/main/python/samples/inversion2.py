from sklearn.cross_validation import StratifiedKFold, KFold, ShuffleSplit,train_test_split, PredefinedSplit
from sklearn.ensemble import RandomForestRegressor , ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from scipy.stats import randint, uniform
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from gini import normalized_gini

# Normalized Gini Scorer
gini_scorer = metrics.make_scorer(normalized_gini, greater_is_better = True)


dat=pd.read_table('../input/train.csv',sep=",")
y=dat[['Hazard']].values.ravel()
dat=dat.drop(['Hazard','Id'],axis=1)

factors = train.select_dtypes(include=['object']).columns


folds=train_test_split(range(len(y)),test_size=0.30, random_state=15) #30% test

#First one hot and make a pandas df
dat_dict=dat.T.to_dict().values()
vectorizer = DV( sparse = False )
vectorizer.fit( dat_dict )
dat= vectorizer.transform( dat_dict )
dat=pd.DataFrame(dat)


train_X=dat.iloc[folds[0],:]
train_y=y[folds[0]]
test_X=dat.iloc[folds[1],:]
test_y=y[folds[1]]


rf=RandomForestRegressor(n_estimators=500, n_jobs=1, random_state=15)
rf.fit(train_X,train_y)
y_submission=rf.predict(test_X)
print("Validation Sample Score: {:.10f} (normalized gini).".format(normalized_gini(test_y,y_submission)))
