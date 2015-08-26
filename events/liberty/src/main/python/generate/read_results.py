import os.path
import pandas as pd
import numpy as np
LOC_BASE = '/Users/rbekbolatov/tmp/runaug25'
LOC_RESULTS = LOC_BASE + '/results'
LOC_TASKS = LOC_BASE + '/tasks'

TASK_OFFSET=0

# show
finished_locs = [
    LOC_RESULTS + '/TASK_' + str(idx + TASK_OFFSET)
    for idx in range(1800)
    if os.path.isfile(LOC_RESULTS + '/TASK_' + str(idx+TASK_OFFSET) + '/task_done')
    ]

results_blended = pd.DataFrame()
for loc in finished_locs:
    argsfile = open(loc + '/args.txt', 'r')
    args = argsfile.readline()[:-1]
    argsfile.close()
    rb = pd.read_csv(loc + '/results_blended.csv')
    results_blended[args] = rb['0']

#results_blended.mean().order()[-30:]
results_blended = results_blended.reindex_axis(results_blended.mean().order().index, axis=1)
results_blended = results_blended.iloc[:, -30:]
results_blended.boxplot(vert=False, showmeans=True)


results = pd.DataFrame()
for loc in finished_locs:
    argsfile = open(loc + '/args.txt', 'r')
    args = argsfile.readline()[:-1]
    argsfile.close()
    rb = pd.read_csv(loc + '/results.csv')
    print(args)
    print(rb)
    results[args] = rb['0']

results = results.reindex_axis(results.mean().order().index, axis=1)
results.boxplot(vert=False, showmeans=True)

pd.DataFrame(np.dot(np.asarray(results).T ,np.kron(np.ones((10, 1)), np.eye(5)))).boxplot(showmeans=True)
pd.DataFrame(np.dot(np.asarray(results).T ,np.kron(np.ones((10, 1)), np.eye(5)))).mean()