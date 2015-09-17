import os.path
import pandas as pd
import numpy as np

LOC_BASE = '/Users/rbekbolatov/tmp/runsep13'


#ipython
#ec2-52-25-178-60.us-west-2.compute.amazonaws.com
# worker
#ec2-52-27-130-165.us-west-2.compute.amazonaws.com

hosts = [
    "ec2-52-89-109-187.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-170.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-184.us-west-2.compute.amazonaws.com",
    "ec2-52-11-209-125.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-202.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-61.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-146.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-179.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-182.us-west-2.compute.amazonaws.com",
    "ec2-52-25-58-77.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-195.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-180.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-110.us-west-2.compute.amazonaws.com",
    "ec2-52-24-26-170.us-west-2.compute.amazonaws.com",
    "ec2-52-89-86-231.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-122.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-167.us-west-2.compute.amazonaws.com",
    "ec2-52-88-244-17.us-west-2.compute.amazonaws.com",
    "ec2-52-89-85-182.us-west-2.compute.amazonaws.com",
    "ec2-52-88-205-245.us-west-2.compute.amazonaws.com",
    "ec2-52-89-98-56.us-west-2.compute.amazonaws.com",
    "ec2-52-27-179-76.us-west-2.compute.amazonaws.com",
    "ec2-52-89-95-232.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-159.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-117.us-west-2.compute.amazonaws.com",
    "ec2-52-89-83-74.us-west-2.compute.amazonaws.com",
    "ec2-52-89-103-159.us-west-2.compute.amazonaws.com",
    "ec2-52-27-130-165.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-177.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-176.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-150.us-west-2.compute.amazonaws.com",
    "ec2-52-11-123-225.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-213.us-west-2.compute.amazonaws.com",
    "ec2-52-89-31-158.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-174.us-west-2.compute.amazonaws.com",
    "ec2-52-89-37-106.us-west-2.compute.amazonaws.com",
    "ec2-52-88-192-88.us-west-2.compute.amazonaws.com",
    "ec2-52-89-102-179.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-196.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-192.us-west-2.compute.amazonaws.com",
    "ec2-52-27-87-249.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-216.us-west-2.compute.amazonaws.com",
    "ec2-52-89-33-170.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-211.us-west-2.compute.amazonaws.com",
    "ec2-52-89-78-197.us-west-2.compute.amazonaws.com",
    "ec2-52-89-108-31.us-west-2.compute.amazonaws.com",
    "ec2-52-89-47-29.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-156.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-121.us-west-2.compute.amazonaws.com",
    "ec2-52-88-66-252.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-140.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-183.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-59.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-208.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-171.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-142.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-206.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-181.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-201.us-west-2.compute.amazonaws.com",
    "ec2-52-89-98-159.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-197.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-234.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-66.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-102.us-west-2.compute.amazonaws.com",
    "ec2-52-89-91-43.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-188.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-135.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-186.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-63.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-194.us-west-2.compute.amazonaws.com",
    "ec2-52-89-109-113.us-west-2.compute.amazonaws.com"
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


