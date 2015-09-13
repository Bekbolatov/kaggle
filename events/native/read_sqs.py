import boto3
import requests
from parser_runner import ParserRunner

parser = ParserRunner()

dns_name_req = requests.get('http://169.254.169.254/latest/meta-data/public-hostname')
dns_name = dns_name_req.text

sqs = boto3.resource('sqs')
queue = sqs.get_queue_by_name(QueueName='cluster_task_queue')
new_member_queue = sqs.get_queue_by_name(QueueName='cluster_new_member')
commitment_queue = sqs.get_queue_by_name(QueueName='cluster_task_commitment')

def send_dns_to_queue(receiving_queue, msg):
    receiving_queue.send_message(MessageBody=msg, MessageAttributes={
        'Worker': {
            'StringValue': dns_name,
            'DataType': 'String'
            }
        })

def send_to_member_queue(msg):
    send_dns_to_queue(new_member_queue, msg)

def send_to_commit_queue(msg):
    send_dns_to_queue(commitment_queue, msg)

def log(msg):
    with open('/var/log/renat_cluster/daemon.log', 'a') as f:
        f.write(msg + '\n')


send_to_member_queue('new_member')

keep_receiving = True
while keep_receiving:
    for message in queue.receive_messages():
        msg = message.body
        log('Received: {0}'.format(msg))
        if msg == 'quit':
            keep_receiving = False
            log('Quitting')
            send_to_member_queue('exit_member')
            break
        if msg.startswith('parse:'):
            send_to_commit_queue(msg)
            message.delete()
            log('Processing parsing task: {0}'.format(msg))
            parser.run(msg[6:])


