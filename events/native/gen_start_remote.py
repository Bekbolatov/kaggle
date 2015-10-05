import os.path
import pandas as pd
import numpy as np

LOC_BASE = '/Users/rbekbolatov/tmp/runsep13'


#ipython
#ec2-52-25-178-60.us-west-2.compute.amazonaws.com
# worker
#ec2-52-27-130-165.us-west-2.compute.amazonaws.com

hosts = [
    "ec2-52-89-193-15.us-west-2.compute.amazonaws.com",
    "ec2-54-69-136-78.us-west-2.compute.amazonaws.com",
    "ec2-54-68-193-159.us-west-2.compute.amazonaws.com",
    "ec2-52-88-201-154.us-west-2.compute.amazonaws.com",
    "ec2-52-89-15-13.us-west-2.compute.amazonaws.com",
    "ec2-52-89-45-252.us-west-2.compute.amazonaws.com",
    "ec2-54-69-129-216.us-west-2.compute.amazonaws.com",
    "ec2-52-89-235-49.us-west-2.compute.amazonaws.com",
    "ec2-54-68-201-203.us-west-2.compute.amazonaws.com",
    "ec2-52-10-43-242.us-west-2.compute.amazonaws.com",
    "ec2-52-88-239-14.us-west-2.compute.amazonaws.com",
    "ec2-54-69-13-4.us-west-2.compute.amazonaws.com",
    "ec2-52-89-133-116.us-west-2.compute.amazonaws.com",
    "ec2-54-68-203-141.us-west-2.compute.amazonaws.com",
    "ec2-54-69-128-40.us-west-2.compute.amazonaws.com",
    "ec2-52-89-243-103.us-west-2.compute.amazonaws.com",
    "ec2-54-68-167-111.us-west-2.compute.amazonaws.com",
    "ec2-54-68-149-49.us-west-2.compute.amazonaws.com",
    "ec2-52-88-140-31.us-west-2.compute.amazonaws.com",
    "ec2-54-69-13-17.us-west-2.compute.amazonaws.com",
    "ec2-54-68-108-253.us-west-2.compute.amazonaws.com",
    "ec2-52-89-135-95.us-west-2.compute.amazonaws.com",
    "ec2-52-89-221-244.us-west-2.compute.amazonaws.com",
    "ec2-52-89-254-95.us-west-2.compute.amazonaws.com",
    "ec2-54-68-166-47.us-west-2.compute.amazonaws.com",
    "ec2-54-69-43-170.us-west-2.compute.amazonaws.com",
    "ec2-52-89-151-225.us-west-2.compute.amazonaws.com",
    "ec2-54-69-133-194.us-west-2.compute.amazonaws.com",
    "ec2-52-26-206-14.us-west-2.compute.amazonaws.com",
    "ec2-52-88-57-125.us-west-2.compute.amazonaws.com",
    "ec2-54-69-102-121.us-west-2.compute.amazonaws.com",
    "ec2-52-88-5-19.us-west-2.compute.amazonaws.com",
    "ec2-52-25-5-100.us-west-2.compute.amazonaws.com",
    "ec2-54-68-125-99.us-west-2.compute.amazonaws.com",
    "ec2-52-89-55-24.us-west-2.compute.amazonaws.com",
    "ec2-54-68-179-101.us-west-2.compute.amazonaws.com",
    "ec2-54-69-7-70.us-west-2.compute.amazonaws.com",
    "ec2-54-68-46-25.us-west-2.compute.amazonaws.com",
    "ec2-52-27-101-119.us-west-2.compute.amazonaws.com",
    "ec2-54-69-16-198.us-west-2.compute.amazonaws.com",
    "ec2-52-27-202-38.us-west-2.compute.amazonaws.com",
    "ec2-52-89-241-245.us-west-2.compute.amazonaws.com",
    "ec2-52-89-188-228.us-west-2.compute.amazonaws.com",
    "ec2-54-68-163-25.us-west-2.compute.amazonaws.com",
    "ec2-54-69-100-163.us-west-2.compute.amazonaws.com",
    "ec2-52-89-215-11.us-west-2.compute.amazonaws.com",
    "ec2-52-89-93-129.us-west-2.compute.amazonaws.com",
    "ec2-54-68-148-244.us-west-2.compute.amazonaws.com",
    "ec2-54-68-205-216.us-west-2.compute.amazonaws.com",
    "ec2-54-69-111-80.us-west-2.compute.amazonaws.com",
    "ec2-54-69-16-189.us-west-2.compute.amazonaws.com",
    "ec2-54-69-13-59.us-west-2.compute.amazonaws.com",
    "ec2-52-89-248-201.us-west-2.compute.amazonaws.com",
    "ec2-52-88-167-15.us-west-2.compute.amazonaws.com",
    "ec2-54-68-102-238.us-west-2.compute.amazonaws.com",
    "ec2-54-68-197-197.us-west-2.compute.amazonaws.com",
    "ec2-52-24-252-202.us-west-2.compute.amazonaws.com",
    "ec2-52-26-56-255.us-west-2.compute.amazonaws.com",
    "ec2-52-11-188-93.us-west-2.compute.amazonaws.com",
    "ec2-52-89-37-225.us-west-2.compute.amazonaws.com",
    "ec2-54-69-134-170.us-west-2.compute.amazonaws.com",
    "ec2-54-68-230-132.us-west-2.compute.amazonaws.com",
    "ec2-52-26-247-58.us-west-2.compute.amazonaws.com",
    "ec2-54-69-22-62.us-west-2.compute.amazonaws.com",
    "ec2-52-88-148-10.us-west-2.compute.amazonaws.com",
    "ec2-54-68-104-52.us-west-2.compute.amazonaws.com",
    "ec2-54-68-115-107.us-west-2.compute.amazonaws.com",
    "ec2-52-89-241-191.us-west-2.compute.amazonaws.com",
    "ec2-54-69-136-58.us-west-2.compute.amazonaws.com",
    "ec2-54-69-11-201.us-west-2.compute.amazonaws.com",
    "ec2-52-10-114-136.us-west-2.compute.amazonaws.com",
    "ec2-54-68-202-216.us-west-2.compute.amazonaws.com",
    "ec2-54-69-136-34.us-west-2.compute.amazonaws.com",
    "ec2-52-88-34-128.us-west-2.compute.amazonaws.com",
    "ec2-52-88-176-236.us-west-2.compute.amazonaws.com",
    "ec2-52-89-208-176.us-west-2.compute.amazonaws.com",
    "ec2-52-24-102-152.us-west-2.compute.amazonaws.com",
    "ec2-52-89-227-161.us-west-2.compute.amazonaws.com",
    "ec2-54-68-203-253.us-west-2.compute.amazonaws.com",
    "ec2-52-89-90-171.us-west-2.compute.amazonaws.com",
    "ec2-52-89-196-70.us-west-2.compute.amazonaws.com",
    "ec2-52-88-12-15.us-west-2.compute.amazonaws.com",
    "ec2-52-89-175-90.us-west-2.compute.amazonaws.com",
    "ec2-52-89-230-26.us-west-2.compute.amazonaws.com",
    "ec2-54-68-156-176.us-west-2.compute.amazonaws.com",
    "ec2-52-89-163-39.us-west-2.compute.amazonaws.com",
    "ec2-54-69-112-242.us-west-2.compute.amazonaws.com",
    "ec2-52-25-123-5.us-west-2.compute.amazonaws.com",
    "ec2-52-88-216-208.us-west-2.compute.amazonaws.com",
    "ec2-52-25-242-153.us-west-2.compute.amazonaws.com",
    "ec2-54-68-131-38.us-west-2.compute.amazonaws.com",
    "ec2-52-88-135-209.us-west-2.compute.amazonaws.com",
    "ec2-52-27-6-21.us-west-2.compute.amazonaws.com",
    "ec2-54-69-136-66.us-west-2.compute.amazonaws.com",
    "ec2-52-89-222-49.us-west-2.compute.amazonaws.com",
    "ec2-52-89-226-215.us-west-2.compute.amazonaws.com",
    "ec2-52-89-187-92.us-west-2.compute.amazonaws.com",
    "ec2-52-24-99-129.us-west-2.compute.amazonaws.com",
    "ec2-52-89-132-66.us-west-2.compute.amazonaws.com",
    "ec2-52-89-73-242.us-west-2.compute.amazonaws.com"
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

task_sender = open(LOC_BASE + '/cat_daemon.sh', 'w')
task_sender.write("#!/bin/bash\n")
for host in hosts:
    task_sender.write('ssh ' + host + ' \'cat /home/ec2-user/logs/daemon.log\'\n')
task_sender.close()


