import boto3
import requests

dns_name_req = requests.get('http://169.254.169.254/latest/meta-data/public-hostname')
dns_name = dns_name_req.text

sqs = boto3.resource('sqs')
queue = sqs.get_queue_by_name(QueueName='cluster_task_queue')

new_member_queue = sqs.get_queue_by_name(QueueName='cluster_new_member')
new_member_queue.send_message(MessageBody='new_member', MessageAttributes={
    'Worker': {
        'StringValue': dns_name,
        'DataType': 'String'
        }
    })

commitment_queue = sqs.get_queue_by_name(QueueName='cluster_task_commitment')

keep_receiving = True
while keep_receiving:
    for message in queue.receive_messages():
        msg = message.body
        with open('/var/log/renat_cluster/daemon.log', 'a') as f:
            f.write('Received: {0}\n'.format(msg))
        commitment_queue.send_message(MessageBody=msg, MessageAttributes={
            'Worker': {
                'StringValue': dns_name,
                'DataType': 'String'
                }
            })
        message.delete()
        if msg == 'quit':
            keep_receiving = False
            break


