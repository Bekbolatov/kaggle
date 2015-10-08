import os.path
import pandas as pd
import numpy as np

LOC_BASE = '/Users/rbekbolatov/tmp/runsep13'


#ipython
#ec2-52-25-178-60.us-west-2.compute.amazonaws.com
# worker
#ec2-52-27-130-165.us-west-2.compute.amazonaws.com

hosts = [u'ec2-54-68-162-217.us-west-2.compute.amazonaws.com',
         u'ec2-54-68-197-7.us-west-2.compute.amazonaws.com',
         u'ec2-54-68-171-216.us-west-2.compute.amazonaws.com',
         u'ec2-54-68-150-164.us-west-2.compute.amazonaws.com',
         u'ec2-54-68-146-117.us-west-2.compute.amazonaws.com',
         u'ec2-54-68-152-53.us-west-2.compute.amazonaws.com',
         u'ec2-52-27-62-213.us-west-2.compute.amazonaws.com',
         u'ec2-52-26-107-31.us-west-2.compute.amazonaws.com',
         u'ec2-54-68-182-104.us-west-2.compute.amazonaws.com',
         u'ec2-52-88-92-189.us-west-2.compute.amazonaws.com',
         u'ec2-52-88-108-200.us-west-2.compute.amazonaws.com',
         u'ec2-54-68-202-212.us-west-2.compute.amazonaws.com',
         u'ec2-54-68-171-249.us-west-2.compute.amazonaws.com',
         u'ec2-54-68-190-5.us-west-2.compute.amazonaws.com',
         u'ec2-54-68-150-149.us-west-2.compute.amazonaws.com',
         u'ec2-52-88-86-78.us-west-2.compute.amazonaws.com',
         u'ec2-52-88-80-195.us-west-2.compute.amazonaws.com',
         u'ec2-52-24-33-69.us-west-2.compute.amazonaws.com']

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

import xgboost

xgboost.XGBModel


xgboost.Booster
