import pandas as pd
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.cross_validation import train_test_split

from gini import normalized_gini, gini_eval
from dataset import get_data

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


params = pd.DataFrame({
    "objective": "reg:linear",
    "eta": [0.04, 0.03, 0.03, 0.03, 0.02],
    "min_child_weight": 5,
    "subsample": [1, 0.9, 0.95, 1, 0.6],
    "colsample_bytree": [0.7, 0.6, 0.65, 0.6, 0.85],
    "max_depth": [8, 7, 9, 10, 10],
    "eval_metric": "auc",
    "scale_pos_weight": 1,
    "silent": 1
})


dat_x, dat_y, lb_x, lb_ind = get_data()

train_index, test_index = train_test_split(range(dat_x.shape[0]), test_size=0.1, random_state=102)

train_x = dat_x.iloc[train_index, :]
train_y = dat_y[train_index]
cv_x = dat_x.iloc[test_index, :]
cv_y = dat_y[test_index]

xgb_train = xgb.DMatrix(train_x, label=train_y)
xgb_cv = xgb.DMatrix(cv_x, label=cv_y)
watchlist = [(xgb_cv, 'cv')]

model = xgb.train(params.iloc[0,:].to_dict(), xgb_train, num_boost_round = 3000,
                  evals = watchlist,
                  feval = gini_eval,
                  verbose_eval = False,
                  early_stopping_rounds=50)

cv_y_preds = model.predict(xgb_cv, ntree_limit=model.best_iteration)
cv_error = normalized_gini(cv_y, cv_y_preds)
print("Validation Sample Score: {:.10f} (normalized gini).".format(cv_error))

preds = pd.DataFrame({"actual": cv_y, "pred": cv_y_preds})
preds.boxplot('pred', 'actual')

# show
# import pylab as pl
# max_cy = max(cv_y) * 1.1
# pl.scatter(cv_y, cv_y_preds, s=1)
# pl.xlim(0, 5)
# pl.show()

# analyze
cv_all = np.hstack( (np.vstack((cv_y, cv_y_preds)).T, cv_x))
dat_all = np.hstack( ( dat_y.reshape(-1, 1), dat_x ))
dat_all_df = pd.DataFrame(dat_all)




# clusters
n_clusters = 2

#data = dat_all_df[dat_all_df[0] > 5].iloc[:, 1:]
#data = dat_all_df.iloc[:, 1:]
data = np.vstack((lb_x, dat_x))

reduced_data = PCA(n_components=2).fit_transform(data)
#plot now
#plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)


#
# from sklearn.neighbors import KernelDensity
# X_plot = np.linspace(reduced_data[:, 0].min(), reduced_data[:, 0].max(), 1000)[:, np.newaxis]
# kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(reduced_data)
# log_dens = kde.score_samples(X_plot)
# fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
# fig.subplots_adjust(hspace=0.05, wspace=0.05)
# ax[1, 0].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')


clustering = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
clustering.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.5     # point in the mesh [x_min, m_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = clustering.predict(np.c_[xx.ravel(), yy.ravel()])



# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = clustering.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()







