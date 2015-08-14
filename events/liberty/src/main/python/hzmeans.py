import pandas as pd
from matplotlib import pylab as plt

train = pd.read_csv('../input/train.csv', index_col=None)

factors = train.select_dtypes(include=['object']).columns
train = train[list(factors) + ['Hazard']]

for feat in factors:
    plt.figure()
    train.boxplot('Hazard', feat)
    plt.title('%s hazard boxplot' % feat)
    plt.ylabel('hazard')
    plt.xticks(rotation=0)
    plt.yscale('log')
    plt.gcf().savefig('%s_hazard.png' % feat)

