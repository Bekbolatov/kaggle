import boto3

class ClusterTaskQueue():
    def add_task(self, task_string):
        sqs = boto3.resource('sqs')
        queue = sqs.get_queue_by_name(QueueName='cluster_task_queue')
        queue.send_message(MessageBody=task_string)

ct = ClusterTaskQueue()
