import os.path
import pandas as pd
import numpy as np
#LOC_BASE = '/home/ec2-user/runaug25_1'
LOC_BASE = '/Users/rbekbolatov/tmp/runaug25'
LOC_RESULTS = LOC_BASE + '/results'
LOC_TASKS = LOC_BASE + '/tasks'

hosts=[
    "ec2-52-11-113-19.us-west-2.compute.amazonaws.com",
    "ec2-54-186-235-125.us-west-2.compute.amazonaws.com",
    "ec2-54-200-168-197.us-west-2.compute.amazonaws.com",
    "ec2-54-186-46-118.us-west-2.compute.amazonaws.com",
    "ec2-54-191-165-16.us-west-2.compute.amazonaws.com",
    "ec2-54-186-219-63.us-west-2.compute.amazonaws.com",
    "ec2-54-201-96-35.us-west-2.compute.amazonaws.com",
    "ec2-54-201-205-138.us-west-2.compute.amazonaws.com",
    "ec2-54-187-89-118.us-west-2.compute.amazonaws.com",
    "ec2-54-201-204-217.us-west-2.compute.amazonaws.com",
    "ec2-54-200-171-138.us-west-2.compute.amazonaws.com",
    "ec2-54-201-194-243.us-west-2.compute.amazonaws.com",
    "ec2-52-26-10-131.us-west-2.compute.amazonaws.com",
    "ec2-52-25-99-129.us-west-2.compute.amazonaws.com",
    "ec2-54-191-64-150.us-west-2.compute.amazonaws.com",
    "ec2-54-201-175-42.us-west-2.compute.amazonaws.com",
    "ec2-54-68-14-181.us-west-2.compute.amazonaws.com",
    "ec2-54-187-122-21.us-west-2.compute.amazonaws.com",
    "ec2-54-201-205-223.us-west-2.compute.amazonaws.com",
    "ec2-54-191-173-239.us-west-2.compute.amazonaws.com",
    "ec2-54-201-211-179.us-west-2.compute.amazonaws.com",
    "ec2-54-201-207-77.us-west-2.compute.amazonaws.com",
    "ec2-54-69-106-129.us-west-2.compute.amazonaws.com",
    "ec2-54-201-215-176.us-west-2.compute.amazonaws.com",
    "ec2-52-27-175-33.us-west-2.compute.amazonaws.com",
    "ec2-54-201-162-196.us-west-2.compute.amazonaws.com",
    "ec2-52-27-31-10.us-west-2.compute.amazonaws.com",
    "ec2-54-187-129-99.us-west-2.compute.amazonaws.com",
    "ec2-54-201-211-126.us-west-2.compute.amazonaws.com",
    "ec2-54-201-204-201.us-west-2.compute.amazonaws.com",
    "ec2-54-148-117-139.us-west-2.compute.amazonaws.com",
    "ec2-54-201-204-213.us-west-2.compute.amazonaws.com",
    "ec2-52-24-221-10.us-west-2.compute.amazonaws.com",
    "ec2-54-191-238-253.us-west-2.compute.amazonaws.com",
    "ec2-54-69-77-56.us-west-2.compute.amazonaws.com",
    "ec2-54-201-194-86.us-west-2.compute.amazonaws.com",
    "ec2-54-201-188-252.us-west-2.compute.amazonaws.com",
    "ec2-52-27-118-206.us-west-2.compute.amazonaws.com",
    "ec2-52-24-192-247.us-west-2.compute.amazonaws.com",
    "ec2-52-24-36-201.us-west-2.compute.amazonaws.com",
    "ec2-54-201-210-115.us-west-2.compute.amazonaws.com",
]

TASK_OFFSET = 1200
tasks = list(enumerate(['0:7;' + str(d) for d in range(32)] +
                       [str(a) + ':' + str(b) + ',0:7;'
                        for a in range(32)
                        for b in range(a + 1, 32) if (a,b) != (0,7)], start=TASK_OFFSET))
num_hosts = len(hosts)
num_tasks = len(tasks)
host_tasks = [(host, np.array(tasks)[range(i, num_tasks, num_hosts)]) for i, host in enumerate(hosts)]


# distrib_keys = open(LOC_BASE + '/distrib_keys.sh', 'w')
# for host in hosts:
#     distrib_keys.write('scp /Users/rbekbolatov/repos/gh/bekbolatov/kaggle/tmp/authorized_keys ' + host + ':/home/ec2-user/.ssh/authorized_keys\n')
# distrib_keys.close()

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
    task_sender.write('scp ' + tasks_file_loc + ' ' + host + ':/home/ec2-user/input_queue\n')
    # results receiving
    for task_num, _ in ts:
        task_directory = '/TASK_' + str(task_num)
        location = LOC_RESULTS + task_directory
        results_receiver.write('if [[ ! -e "' + location + '/task_done" ]]; then scp -r ' + host + ':/home/ec2-user' + task_directory + ' ' + LOC_RESULTS + '/. ; fi\n')
task_sender.close()
results_receiver.close()

# start queues
task_sender = open(LOC_BASE + '/start_tasks.sh', 'w')
task_sender.write("#!/bin/bash\n")
for host in hosts:
    tasks_file_loc = LOC_BASE + '/tasks/' + host
    task_sender.write('ssh ' + host + ' /home/ec2-user/runscript_dropcol.sh & sleep 1\n')
task_sender.close()

