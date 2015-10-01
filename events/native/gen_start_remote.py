import os.path
import pandas as pd
import numpy as np

LOC_BASE = '/Users/rbekbolatov/tmp/runsep13'


#ipython
#ec2-52-25-178-60.us-west-2.compute.amazonaws.com
# worker
#ec2-52-27-130-165.us-west-2.compute.amazonaws.com

hosts = [
    "ec2-52-89-29-186.us-west-2.compute.amazonaws.com",
    "ec2-52-89-198-199.us-west-2.compute.amazonaws.com",
    "ec2-52-89-197-16.us-west-2.compute.amazonaws.com",
    "ec2-52-89-198-190.us-west-2.compute.amazonaws.com",
    "ec2-52-11-167-24.us-west-2.compute.amazonaws.com",
    "ec2-52-89-133-158.us-west-2.compute.amazonaws.com",
    "ec2-52-88-216-74.us-west-2.compute.amazonaws.com",
    "ec2-52-89-16-0.us-west-2.compute.amazonaws.com",
    "ec2-52-89-173-146.us-west-2.compute.amazonaws.com",
    "ec2-52-89-201-45.us-west-2.compute.amazonaws.com",
    "ec2-52-89-196-247.us-west-2.compute.amazonaws.com",
    "ec2-52-25-220-1.us-west-2.compute.amazonaws.com",
    "ec2-52-24-119-81.us-west-2.compute.amazonaws.com",
    "ec2-52-89-169-123.us-west-2.compute.amazonaws.com",
    "ec2-52-89-199-105.us-west-2.compute.amazonaws.com",
    "ec2-52-88-193-167.us-west-2.compute.amazonaws.com",
    "ec2-52-89-200-229.us-west-2.compute.amazonaws.com",
    "ec2-52-89-200-246.us-west-2.compute.amazonaws.com",
    "ec2-52-89-201-66.us-west-2.compute.amazonaws.com",
    "ec2-52-89-200-65.us-west-2.compute.amazonaws.com",
    "ec2-52-89-198-232.us-west-2.compute.amazonaws.com",
    "ec2-52-25-89-202.us-west-2.compute.amazonaws.com",
    "ec2-52-89-200-186.us-west-2.compute.amazonaws.com",
    "ec2-52-89-199-77.us-west-2.compute.amazonaws.com",
    "ec2-52-88-221-49.us-west-2.compute.amazonaws.com",
    "ec2-52-89-139-102.us-west-2.compute.amazonaws.com",
    "ec2-52-25-238-186.us-west-2.compute.amazonaws.com",
    "ec2-52-11-138-168.us-west-2.compute.amazonaws.com",
    "ec2-52-89-144-242.us-west-2.compute.amazonaws.com",
    "ec2-52-89-199-102.us-west-2.compute.amazonaws.com",
    "ec2-52-89-141-25.us-west-2.compute.amazonaws.com",
    "ec2-52-89-197-40.us-west-2.compute.amazonaws.com",
    "ec2-52-89-199-129.us-west-2.compute.amazonaws.com",
    "ec2-52-89-87-234.us-west-2.compute.amazonaws.com",
    "ec2-52-10-135-33.us-west-2.compute.amazonaws.com",
    "ec2-52-89-151-153.us-west-2.compute.amazonaws.com",
    "ec2-52-89-198-202.us-west-2.compute.amazonaws.com",
    "ec2-52-89-139-115.us-west-2.compute.amazonaws.com",
    "ec2-52-89-201-168.us-west-2.compute.amazonaws.com",
    "ec2-52-89-199-82.us-west-2.compute.amazonaws.com",
    "ec2-52-89-144-226.us-west-2.compute.amazonaws.com",
    "ec2-52-88-125-122.us-west-2.compute.amazonaws.com",
    "ec2-52-89-147-190.us-west-2.compute.amazonaws.com",
    "ec2-52-89-128-50.us-west-2.compute.amazonaws.com",
    "ec2-52-89-196-159.us-west-2.compute.amazonaws.com",
    "ec2-52-89-199-1.us-west-2.compute.amazonaws.com",
    "ec2-52-89-201-29.us-west-2.compute.amazonaws.com",
    "ec2-52-26-46-211.us-west-2.compute.amazonaws.com",
    "ec2-52-89-201-73.us-west-2.compute.amazonaws.com",
    "ec2-52-89-197-227.us-west-2.compute.amazonaws.com",
    "ec2-52-89-201-173.us-west-2.compute.amazonaws.com",
    "ec2-52-89-180-132.us-west-2.compute.amazonaws.com",
    "ec2-52-89-176-205.us-west-2.compute.amazonaws.com",
    "ec2-52-89-200-141.us-west-2.compute.amazonaws.com",
    "ec2-52-89-199-194.us-west-2.compute.amazonaws.com",
    "ec2-52-89-200-119.us-west-2.compute.amazonaws.com",
    "ec2-52-89-199-71.us-west-2.compute.amazonaws.com",
    "ec2-52-88-113-50.us-west-2.compute.amazonaws.com",
    "ec2-52-89-200-77.us-west-2.compute.amazonaws.com",
    "ec2-52-89-139-194.us-west-2.compute.amazonaws.com",
    "ec2-52-89-101-94.us-west-2.compute.amazonaws.com",
    "ec2-52-89-199-43.us-west-2.compute.amazonaws.com",
    "ec2-52-89-201-198.us-west-2.compute.amazonaws.com",
    "ec2-52-89-197-39.us-west-2.compute.amazonaws.com",
    "ec2-52-88-232-199.us-west-2.compute.amazonaws.com",
    "ec2-52-24-93-225.us-west-2.compute.amazonaws.com",
    "ec2-52-89-201-230.us-west-2.compute.amazonaws.com",
    "ec2-52-89-197-9.us-west-2.compute.amazonaws.com",
    "ec2-52-89-195-142.us-west-2.compute.amazonaws.com",
    "ec2-52-88-51-55.us-west-2.compute.amazonaws.com",
    "ec2-52-89-199-81.us-west-2.compute.amazonaws.com",
    "ec2-52-89-138-194.us-west-2.compute.amazonaws.com",
    "ec2-52-27-136-75.us-west-2.compute.amazonaws.com",
    "ec2-52-89-110-144.us-west-2.compute.amazonaws.com",
    "ec2-52-89-199-60.us-west-2.compute.amazonaws.com",
    "ec2-52-89-201-228.us-west-2.compute.amazonaws.com",
    "ec2-52-89-200-136.us-west-2.compute.amazonaws.com",
    "ec2-52-89-201-74.us-west-2.compute.amazonaws.com",
    "ec2-52-89-189-206.us-west-2.compute.amazonaws.com",
    "ec2-52-89-36-70.us-west-2.compute.amazonaws.com",
    "ec2-52-89-198-233.us-west-2.compute.amazonaws.com",
    "ec2-52-89-201-3.us-west-2.compute.amazonaws.com",
    "ec2-52-89-200-191.us-west-2.compute.amazonaws.com",
    "ec2-52-89-152-168.us-west-2.compute.amazonaws.com",
    "ec2-52-89-201-79.us-west-2.compute.amazonaws.com",
    "ec2-52-89-200-133.us-west-2.compute.amazonaws.com",
    "ec2-52-89-113-175.us-west-2.compute.amazonaws.com",
    "ec2-52-89-139-207.us-west-2.compute.amazonaws.com",
    "ec2-52-89-108-209.us-west-2.compute.amazonaws.com",
    "ec2-52-89-200-205.us-west-2.compute.amazonaws.com",
    "ec2-52-89-198-240.us-west-2.compute.amazonaws.com",
    "ec2-52-89-45-156.us-west-2.compute.amazonaws.com"
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


