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
    for idx in range(600)
    if os.path.isfile(LOC_RESULTS + '/TASK_' + str(idx+TASK_OFFSET) + '/task_done')
    ]

results_blended = pd.DataFrame()
for loc in finished_locs:
    argsfile = open(loc + '/args.txt', 'r')
    args = argsfile.readline()[:-1]
    argsfile.close()
    rb = pd.read_csv(loc + '/results_blended.csv')
    print(args)
    print(rb)
    results_blended[args] = rb['0']

results_blended.mean().order()[-20:]
results_blended = results_blended.reindex_axis(results_blended.mean().order().index, axis=1)
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
