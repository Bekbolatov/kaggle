import os.path
import pandas as pd
import numpy as np

LOC_BASE = '/Users/rbekbolatov/tmp/runsep13'

hosts = [
    "ec2-52-89-10-42.us-west-2.compute.amazonaws.com",
    "ec2-52-88-240-67.us-west-2.compute.amazonaws.com",
    "ec2-52-89-2-13.us-west-2.compute.amazonaws.com",
    "ec2-52-89-11-170.us-west-2.compute.amazonaws.com",
    "ec2-52-89-11-193.us-west-2.compute.amazonaws.com",
    "ec2-52-89-7-152.us-west-2.compute.amazonaws.com",
    "ec2-52-89-5-149.us-west-2.compute.amazonaws.com",
    "ec2-52-26-26-62.us-west-2.compute.amazonaws.com",
    "ec2-52-89-11-195.us-west-2.compute.amazonaws.com",
    "ec2-52-89-11-32.us-west-2.compute.amazonaws.com",
    "ec2-52-88-250-249.us-west-2.compute.amazonaws.com",
    "ec2-52-88-254-166.us-west-2.compute.amazonaws.com",
    "ec2-52-89-2-4.us-west-2.compute.amazonaws.com",
    "ec2-52-88-254-121.us-west-2.compute.amazonaws.com",
    "ec2-52-88-248-214.us-west-2.compute.amazonaws.com",
    "ec2-52-89-2-24.us-west-2.compute.amazonaws.com",
    "ec2-52-88-254-160.us-west-2.compute.amazonaws.com",
    "ec2-52-88-243-239.us-west-2.compute.amazonaws.com",
    "ec2-52-89-1-250.us-west-2.compute.amazonaws.com",
    "ec2-52-89-11-189.us-west-2.compute.amazonaws.com",
    "ec2-52-88-215-233.us-west-2.compute.amazonaws.com",
    "ec2-52-88-238-161.us-west-2.compute.amazonaws.com",
    "ec2-52-89-1-202.us-west-2.compute.amazonaws.com",
    "ec2-52-89-9-126.us-west-2.compute.amazonaws.com",
    "ec2-52-89-11-168.us-west-2.compute.amazonaws.com",
    "ec2-52-88-254-150.us-west-2.compute.amazonaws.com",
    "ec2-52-89-8-213.us-west-2.compute.amazonaws.com",
    "ec2-52-89-11-164.us-west-2.compute.amazonaws.com",
    "ec2-52-88-249-97.us-west-2.compute.amazonaws.com",
    "ec2-52-89-10-150.us-west-2.compute.amazonaws.com",
    "ec2-52-88-240-4.us-west-2.compute.amazonaws.com",
    "ec2-52-88-255-106.us-west-2.compute.amazonaws.com",
    "ec2-52-89-11-41.us-west-2.compute.amazonaws.com",
    "ec2-52-89-8-82.us-west-2.compute.amazonaws.com",
    "ec2-52-88-248-53.us-west-2.compute.amazonaws.com",
    "ec2-52-88-246-97.us-west-2.compute.amazonaws.com",
    "ec2-52-27-243-204.us-west-2.compute.amazonaws.com",
    "ec2-52-88-104-26.us-west-2.compute.amazonaws.com",
    "ec2-52-89-9-244.us-west-2.compute.amazonaws.com",
    "ec2-52-88-215-242.us-west-2.compute.amazonaws.com",
    "ec2-52-89-9-125.us-west-2.compute.amazonaws.com",
    "ec2-52-88-249-93.us-west-2.compute.amazonaws.com",
    "ec2-52-88-215-234.us-west-2.compute.amazonaws.com",
    "ec2-52-26-213-218.us-west-2.compute.amazonaws.com",
    "ec2-52-88-139-148.us-west-2.compute.amazonaws.com",
    "ec2-52-88-253-8.us-west-2.compute.amazonaws.com",
    "ec2-52-89-1-48.us-west-2.compute.amazonaws.com",
    "ec2-52-88-226-124.us-west-2.compute.amazonaws.com",
    "ec2-52-89-9-127.us-west-2.compute.amazonaws.com",
    "ec2-52-89-2-12.us-west-2.compute.amazonaws.com",
    "ec2-52-88-254-162.us-west-2.compute.amazonaws.com"
    ]

num_hosts = len(hosts)


task_sender = open(LOC_BASE + '/start_tasks.sh', 'w')
task_sender.write("#!/bin/bash\n")
for host in hosts:
    tasks_file_loc = LOC_BASE + '/tasks/' + host
    task_sender.write('ssh ' + host + ' -t \'tmux new-session -d -s server-session "python /home/ec2-user/repos/bekbolatov/kaggle/events/native/read_sqs.py"\'\n')
task_sender.close()


task_sender = open(LOC_BASE + '/repo_pull.sh', 'w')
task_sender.write("#!/bin/bash\n")
for host in hosts:
    tasks_file_loc = LOC_BASE + '/tasks/' + host
    task_sender.write('ssh ' + host + ' \'cd /home/ec2-user/repos/bekbolatov/kaggle; git pull\'\n')
task_sender.close()


