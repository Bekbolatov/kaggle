
ssh ec2-52-88-104-26.us-west-2.compute.amazonaws.com -t 'tmux new-session -d -s server-session "python /home/ec2-user/repos/bekbolatov/kaggle/events/native/read_sqs.py"'


