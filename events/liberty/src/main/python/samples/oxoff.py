import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
from sklearn import pipeline, metrics, grid_search
from sklearn.utils import shuffle


def prepare_data():
    # load train data
    train    = pd.read_csv('../input/train.csv')
    test     = pd.read_csv('../input/test.csv')
    labels   = train.Hazard
    test_ind = test.ix[:,'Id']
    train.drop('Hazard', axis=1, inplace=True)
    train.drop('Id', axis=1, inplace=True)
    test.drop('Id', axis=1, inplace=True)
    train = np.array(train)
    test = np.array(test)
    for i in range(train.shape[1]):
        if type(train[1,i]) is str:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[:,i]) + list(test[:,i]))
            train[:,i] = lbl.transform(train[:,i])
            test[:,i] = lbl.transform(test[:,i])
    return train.astype(float), labels, test.astype(float), test_ind

def test():
    train, labels, _, _ = prepare_data()
    # Create the pipeline
    est = pipeline.Pipeline([
                                #('fs', SelectKBest(score_func=f_classif,k=610)),
                                #('sc', StandardScaler()),
                                ('model', xgb.XGBRegressor()
                                 #XGBoostRegressor2(params=params, offset=5000, num_rounds=2000)
                                )
                            ])


    # Create a parameter grid to search for best parameters for everything in the pipeline
    param_grid = {'model__min_child_weight': [1],
                  'model__subsample': [0.8],
                  'model__max_depth': [5],
                  'model__learning_rate': [0.05],
                  'model__n_estimators': [200]
                  }

    # Normalized Gini Scorer
    gini_scorer = metrics.make_scorer(normalized_gini, greater_is_better = True)

    # Initialize Grid Search Model
    model = grid_search.GridSearchCV(estimator  = est,
                                     param_grid = param_grid,
                                     scoring    = gini_scorer,
                                     verbose    = 10,
                                     n_jobs     = 4,
                                     iid        = True,
                                     refit      = True,
                                     cv         = 2)
    # Fit Grid Search Model
    model.fit(train, labels)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # Get best model
    best_model = model.best_estimator_

    # Fit model with best parameters optimized for normalized_gini
    best_model.fit(train,labels)

    return best_model


def score(model):
    # load test data
    _, _, test, test_ind = prepare_data()
    preds = model.predict(test)
    preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
    preds = preds.set_index('Id')
    preds.to_csv('output.csv')
    return

def main():
    model = test()
    score(model)

#main()

print(normalized_gini(range(10)[::-1], [1]*10))

# test case 1
desired_y = np.array(pd.read_csv("../input/train.csv")["Hazard"])
predicted_y = np.copy(desired_y)
predicted_y[:] = 1
print(normalized_gini(desired_y, predicted_y))
print(normalized_gini(desired_y[::-1], predicted_y))

## test case 2
predicted_y = [i + (i%30) for i in range(100)]
desired_y = [i for i in range(100)]
print(normalized_gini(predicted_y, desired_y))

## test case 3
predicted_y = [i for i in range(100)]
desired_y = [i + (i%30) for i in range(100)]
print(normalized_gini(predicted_y, desired_y))
