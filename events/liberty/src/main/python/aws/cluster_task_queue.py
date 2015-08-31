import boto3

class ClusterTaskQueue():
    def add_task(self, task_string):
        # Get the service resource
        sqs = boto3.resource('sqs')

        # Get the queue. This returns an SQS.Queue instance
        queue = sqs.get_queue_by_name(QueueName='cluster_task_queue')
        queue.send_message(MessageBody=task_string)

        # You can now access identifiers and attributes
        #print(queue.url)
        #print(queue.attributes.get('DelaySeconds'))

        response = queue.send_message(MessageBody=task_string)
        #print(response.get('MessageId'))
        #print(response.get('MD5OfMessageBody'))


        # multiple msg and also attribs
        # queue.send_message(MessageBody='boto3', MessageAttributes={
        #     'Author': {
        #         'StringValue': 'Daniel',
        #         'DataType': 'string'
        #     }
        # })
        #
        # response = queue.send_messages(Entries=[
        #     {
        #         'Id': '1',
        #         'MessageBody': 'world'
        #     },
        #     {
        #         'Id': '2',
        #         'MessageBody': 'boto3',
        #         'MessageAttributes': {
        #             'Author': {
        #                 'StringValue': 'Daniel',
        #                 'DataType': 'string'
        #             }
        #         }
        #     }
        # ])
        #
        # # Print out any failures
        # print(response.get('Failed'))

        # receive
        # for message in queue.receive_messages(MessageAttributeNames=['Author']):
        #     # Get the custom author message attribute if it was set
        #     author_text = ''
        #     if message.message_attributes is not None:
        #         author_name = message.message_attributes.get('Author')
        #         if author_name:
        #             author_text = ' ({0})'.format(author_name)
        #
        #     # Print out the body and author (if set)
        #     print('Hello, {0}!{1}'.format(message.body, author_text))
        #
        #     # Let the queue know that the message is processed
        #     message.delete()




ct = ClusterTaskQueue()



policy = """
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "Stmt1440998563000",
                    "Effect": "Allow",
                    "Action": [
                        "sqs:*"
                    ],
                    "Resource": [
                        "arn:aws:sqs:us-west-2:445803720301:cluster_new_member"
                    ]
                },
                {
                    "Sid": "Stmt1440998929000",
                    "Effect": "Allow",
                    "Action": [
                        "sqs:*"
                    ],
                    "Resource": [
                        "arn:aws:sqs:us-west-2:445803720301:cluster_task_queue"
                    ]
                }
            ]
        }

        """