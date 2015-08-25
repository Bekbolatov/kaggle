import os.path
import pandas as pd
LOC_BASE = '/Users/rbekbolatov/tmp'
LOC = LOC_BASE + '/results'


hosts=[
    "ec2-54-149-238-236.us-west-2.compute.amazonaws.com",
    "ec2-54-200-127-51.us-west-2.compute.amazonaws.com",
    "ec2-54-191-52-131.us-west-2.compute.amazonaws.com",
    "ec2-54-200-153-187.us-west-2.compute.amazonaws.com",
    "ec2-54-186-164-205.us-west-2.compute.amazonaws.com",
    "ec2-54-200-160-143.us-west-2.compute.amazonaws.com",
    "ec2-54-200-114-43.us-west-2.compute.amazonaws.com",
    "ec2-54-200-209-160.us-west-2.compute.amazonaws.com",
    "ec2-54-200-155-99.us-west-2.compute.amazonaws.com",
    "ec2-54-200-58-123.us-west-2.compute.amazonaws.com",
    "ec2-54-200-84-51.us-west-2.compute.amazonaws.com",
    "ec2-54-200-156-74.us-west-2.compute.amazonaws.com",
    "ec2-54-200-148-194.us-west-2.compute.amazonaws.com",
    "ec2-54-149-18-186.us-west-2.compute.amazonaws.com",
    "ec2-54-149-209-148.us-west-2.compute.amazonaws.com",
    "ec2-54-200-111-53.us-west-2.compute.amazonaws.com",
    "ec2-54-200-159-16.us-west-2.compute.amazonaws.com",
    "ec2-54-200-101-214.us-west-2.compute.amazonaws.com",
    "ec2-54-200-36-183.us-west-2.compute.amazonaws.com",
    "ec2-54-200-87-62.us-west-2.compute.amazonaws.com",
    "ec2-54-200-154-21.us-west-2.compute.amazonaws.com",
    "ec2-54-149-44-177.us-west-2.compute.amazonaws.com",
    "ec2-52-25-77-155.us-west-2.compute.amazonaws.com",
    "ec2-52-24-159-64.us-west-2.compute.amazonaws.com",
    "ec2-54-200-161-157.us-west-2.compute.amazonaws.com",
    "ec2-54-200-131-228.us-west-2.compute.amazonaws.com",
    "ec2-52-11-152-144.us-west-2.compute.amazonaws.com",
    "ec2-54-186-181-94.us-west-2.compute.amazonaws.com",
    "ec2-54-200-156-53.us-west-2.compute.amazonaws.com",
    "ec2-54-187-123-157.us-west-2.compute.amazonaws.com",
    "ec2-54-200-168-31.us-west-2.compute.amazonaws.com",
    #"ec2-54-149-238-236.us-west-2.compute.amazonaws.com",  # ONLY WHEN IT IS DONE
]

TASK_OFFSET = 400

# create sending commands
task_sender = open(LOC_BASE + '/send_tasks.sh', 'w')
task_sender.write("#!/bin/bash\n")
for idx, host in enumerate(hosts):
    if idx not in [24]:
        task_sender.write('ssh ' + host + ' \'/home/ec2-user/runscript_dropcol.sh ' + str(idx+TASK_OFFSET) + ' "24:' + str(idx) + ';9,12,23,26,5,19"\' & sleep 1\n')
        # task_sender.write('ssh ' + host + ' \'/home/ec2-user/runscript_dropcol.sh ' + str(idx+TASK_OFFSET) + ' "3:7;' + str(idx) + '"\' & sleep 1\n')
    else:
        # task_sender.write('ssh ' + host + ' \'/home/ec2-user/runscript_dropcol.sh ' + str(idx+TASK_OFFSET) + ' "3:7;' + str(idx) + '"\' & sleep 1\n')
        if idx == 24:
            task_sender.write('ssh ' + host + ' \'/home/ec2-user/runscript_dropcol.sh ' + str(idx+TASK_OFFSET) + ' "24:29;9,12,23,26,5"\' & sleep 1\n')
task_sender.close()

# create receiving commands
results_reader = open(LOC + '/get_results.sh', 'w')
results_reader.write("#!/bin/bash\n")
for idx, host in enumerate(hosts):
    newidx = idx + TASK_OFFSET
    location = 'TASK_' + str(newidx)
    results_reader.write('if [[ ! -e "' + location + '/task_done" ]]; then scp -r ' + host + ':/home/ec2-user/' + location + ' . ; fi\n')
results_reader.close()

# kill
# for idx, host in enumerate(hosts):
#     print('ssh ' + host + ' "ps aux | grep xgboost | grep -v grep | head -n 1 | awk  \'{print \$2}\' | xargs kill" ; sleep 1')



# show results, when ready
TASK_OFFSET = 360
finished = all([os.path.isfile(LOC + '/TASK_' + str(idx + TASK_OFFSET) + '/task_done') for idx, host in enumerate(hosts)])
print("\nCompleted\n" if finished else "\nstill running\n")

#all
finished_locs = [LOC + '/TASK_' + str(idx) for idx in range(560) if os.path.isfile(LOC + '/TASK_' + str(idx) + '/task_done')]

finished_locs = [LOC + '/TASK_' + str(idx + TASK_OFFSET) for idx, host in enumerate(hosts) if os.path.isfile(LOC + '/TASK_' + str(idx+TASK_OFFSET) + '/task_done')]

results_blended = pd.DataFrame()
for loc in finished_locs:
    argsfile = open(loc + '/args.txt', 'r')
    args = argsfile.readline()[:-1]
    argsfile.close()
    rb = pd.read_csv(loc + '/results_blended.csv')
    print(args)
    print(rb)
    results_blended[args] = rb['0']

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
