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
num_hosts = len(hosts)

# create and distribute tasks, receive results
use_mine = open(LOC_BASE + '/use_mine.sh', 'w')
src_file = '/Users/rbekbolatov/repos/gh/bekbolatov/kaggle/events/liberty/src/main/python/xgboost_liberty_stack.py'
dst_file = '/home/ec2-user/repos/bekbolatov/kaggle/events/liberty/src/main/python/xgboost_liberty_stack.py'
use_mine.write("#!/bin/bash\n")
for host in hosts:
    use_mine.write('scp ' + src_file + ' ' + host + ':' + dst_file + '\n')
use_mine.close()
