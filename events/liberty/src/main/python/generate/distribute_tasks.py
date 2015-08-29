import os.path
import pandas as pd
import numpy as np
#LOC_BASE = '/home/ec2-user/runaug25_1'
LOC_BASE = '/Users/rbekbolatov/tmp/runaug28'
LOC_RESULTS = LOC_BASE + '/results'
LOC_TASKS = LOC_BASE + '/tasks'

hosts=[
    "ec2-54-213-163-101.us-west-2.compute.amazonaws.com",
    "ec2-54-187-107-115.us-west-2.compute.amazonaws.com",
    "ec2-54-191-7-46.us-west-2.compute.amazonaws.com",
    "ec2-54-213-118-64.us-west-2.compute.amazonaws.com",
    "ec2-54-200-155-95.us-west-2.compute.amazonaws.com",
    "ec2-54-187-36-57.us-west-2.compute.amazonaws.com",
    "ec2-54-200-210-116.us-west-2.compute.amazonaws.com",
    "ec2-54-213-129-150.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-122.us-west-2.compute.amazonaws.com",
    "ec2-54-191-207-93.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-234.us-west-2.compute.amazonaws.com",
    "ec2-52-26-246-65.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-74.us-west-2.compute.amazonaws.com",
    "ec2-54-213-124-244.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-217.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-118.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-138.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-231.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-69.us-west-2.compute.amazonaws.com",
    "ec2-54-200-88-139.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-99.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-75.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-98.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-123.us-west-2.compute.amazonaws.com",
    "ec2-54-149-204-141.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-127.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-219.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-124.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-140.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-108.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-229.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-114.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-116.us-west-2.compute.amazonaws.com",
    "ec2-54-148-161-65.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-95.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-110.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-215.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-154.us-west-2.compute.amazonaws.com",
    "ec2-54-213-163-201.us-west-2.compute.amazonaws.com",
    "ec2-54-187-12-31.us-west-2.compute.amazonaws.com",
]
num_hosts = len(hosts)

TASK_OFFSET = 300

# explore
# tasks = list(enumerate(['0:7;' + str(d) for d in range(32)] +
#                        [str(a) + ':' + str(b) + ',0:7;'
#                         for a in range(32)
#                         for b in range(a + 1, 32) if (a,b) != (0,7)], start=TASK_OFFSET))

# generate subm
tasks = zip(range(TASK_OFFSET, TASK_OFFSET + num_hosts), ['7:12,14:31,0:3,0:7,0:16,0:28,2:28,10:14,10:27,22:24,24:29,27:28;9,12,23,26']*num_hosts)

#################
num_tasks = len(tasks)
host_tasks = [(host, np.array(tasks)[range(i, num_tasks, num_hosts)]) for i, host in enumerate(hosts)]


# create and distribute tasks, receive results
task_sender = open(LOC_BASE + '/send_tasks.sh', 'w')
results_receiver = open(LOC_BASE + '/receive_results.sh', 'w')
task_sender.write("#!/bin/bash\n")
results_receiver.write("#!/bin/bash\n")
for host, ts in host_tasks:
    # task sending
    tasks_file_loc = LOC_TASKS + '/' + host
    tasks_file = open(tasks_file_loc, 'w')
    tasks_file.write('\n'.join(t[0] + ' ' + t[1] for t in ts) + '\n')
    tasks_file.close()
    task_sender.write('scp -C ' + tasks_file_loc + ' ' + host + ':/home/ec2-user/input_queue\n')
    # results receiving
    for task_num, _ in ts:
        task_directory = '/TASK_' + str(task_num)
        location = LOC_RESULTS + task_directory
        results_receiver.write('if [[ ! -e "' + location + '/task_done" ]]; then scp -rC ' + host + ':/home/ec2-user' + task_directory + ' ' + LOC_RESULTS + '/. ; fi\n')
task_sender.close()
results_receiver.close()

# start queues
task_sender = open(LOC_BASE + '/start_tasks.sh', 'w')
task_sender.write("#!/bin/bash\n")
for host in hosts:
    tasks_file_loc = LOC_BASE + '/tasks/' + host
    task_sender.write('ssh ' + host + ' /home/ec2-user/runscript_dropcol.sh & sleep 1\n')
task_sender.close()


# kill
killer = open(LOC_BASE + '/kill_tasks.sh', 'w')
killer.write("#!/bin/bash\n")
for host in hosts:
    killer.write('ssh ' + host + ' "ps aux | grep xgboost | grep python | grep -v grep | awk \'{print \\$2}\' | xargs kill" \n')
killer.close()


# create and distribute tasks, receive results
use_mine = open(LOC_BASE + '/use_mine.sh', 'w')
src_file = '/Users/rbekbolatov/repos/gh/bekbolatov/kaggle/events/liberty/src/main/python/'
dst_file = '/home/ec2-user/repos/bekbolatov/kaggle/events/liberty/src/main/python/'
use_mine.write("#!/bin/bash\n")
for host in hosts:
    use_mine.write('scp ' + src_file + 'xgboost_liberty_stack.py ' + host + ':' + dst_file + 'xgboost_liberty_stack.py\n')
    use_mine.write('scp ' + src_file + 'dataset.py ' + host + ':' + dst_file + 'dataset.py\n')
use_mine.close()

distrib_keys = open(LOC_BASE + '/distrib_keys.sh', 'w')
for host in hosts:
    distrib_keys.write('scp /Users/rbekbolatov/repos/gh/bekbolatov/kaggle/tmp/authorized_keys ' + host + ':/home/ec2-user/.ssh/authorized_keys\n')
distrib_keys.close()
