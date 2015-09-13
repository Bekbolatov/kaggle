import boto3
import requests
import parser_runner
import git


class Runner:

    def __init__(self):
        self.g = git.cmd.Git('/home/ec2-user/repos/bekbolatov/kaggle')
        self.parser = parser_runner.ParserRunner()

        dns_name_req = requests.get('http://169.254.169.254/latest/meta-data/public-hostname')
        self.dns_name = dns_name_req.text

        sqs = boto3.resource('sqs')
        self.queue = sqs.get_queue_by_name(QueueName='cluster_task_queue')
        self.new_member_queue = sqs.get_queue_by_name(QueueName='cluster_new_member')
        self.commitment_queue = sqs.get_queue_by_name(QueueName='cluster_task_commitment')

        self.send_to_member_queue('new_member')

        self.my_id = None
        self.total_ids = None

    def update_parser(self):
        self.git.pull()
        reload(parser_runner)
        self.parser = parser_runner.ParserRunner()
        


    def send_dns_to_queue(self, receiving_queue, msg):
        receiving_queue.send_message(MessageBody=msg, MessageAttributes={
            'Worker': {
                'StringValue': self.dns_name,
                'DataType': 'String'
                }
            })

    def send_to_member_queue(self, msg):
        self.send_dns_to_queue(self.new_member_queue, msg)

    def send_to_commit_queue(self, msg):
        self.send_dns_to_queue(self.commitment_queue, msg)

    def log(self, msg):
        with open('/home/ec2-user/logs/daemon.log', 'a') as f:
            f.write(msg + '\n')

    def process(self):
        keep_receiving = True
        while keep_receiving:
            for message in self.queue.receive_messages():
                msg = message.body
                self.log('Received: {0}'.format(msg))
                message.delete()
                if msg == 'quit':
                    keep_receiving = False
                    self.log('Quitting')
                    self.send_to_member_queue('exit_member')
                    break
                if msg.startswith('parse:'):
                    self.send_to_commit_queue(msg)
                    message.delete()
                    self.log('Processing parsing task: {0}'.format(msg))
                    input_data = msg[6:]
                    data = input_data.split(':')
                    if (len(data) == 1 and self.my_id and self.total_ids):
                        run_id = data[0]
                    else:
                        run_id, self.my_id, self.total_ids = data
                    self.parser.run(run_id, self.my_id, self.total_ids)
                if msg.startswith('git:pull'):
                    message.delete()
                    self.log('git pull, reload parser')
                    self.update_parser()


if __name__ == '__main__':
    runner = Runner()
    runner.process()







