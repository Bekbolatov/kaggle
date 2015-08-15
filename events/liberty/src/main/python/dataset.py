import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer


def get_data():
    dat = pd.read_table('../input/train.csv', sep=",")
    dat_y = dat[['Hazard']].values.ravel()
    dat = dat.drop(['Hazard', 'Id'], axis=1)

    lb = pd.read_table('../input/test.csv', sep=",")
    lb_indices = lb[['Id']].values.ravel()
    lb = lb.drop(['Id'], axis=1)

    # apply OHE to factor columns
    numerics = dat.select_dtypes(exclude=['object']).columns
    factors = dat.select_dtypes(include=['object']).columns

    dat_numerics = dat.loc[:, numerics]
    dat_factors = dat.loc[:, factors]
    lb_numerics = lb.loc[:, numerics]
    lb_factors = lb.loc[:, factors]

    dat_factors_dict = dat_factors.T.to_dict().values()
    lb_factors_dict = lb_factors.T.to_dict().values()
    vectorizer = DictVectorizer(sparse=False)
    vectorizer.fit(dat_factors_dict)
    vectorizer.fit(lb_factors_dict)
    dat_factors_ohe = vectorizer.transform(dat_factors_dict)
    lb_factors_ohe = vectorizer.transform(lb_factors_dict)

    dat_factors_ohe_df = pd.DataFrame(np.hstack((dat_numerics, dat_factors_ohe)))
    lb_factors_ohe_df = pd.DataFrame(np.hstack((lb_numerics, lb_factors_ohe)))

    return (dat_factors_ohe_df, dat_y, lb_factors_ohe_df, lb_indices)
