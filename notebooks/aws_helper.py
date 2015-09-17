import boto.ec2
import boto3

class SpotInstances:
    """
    si = SpotInstances()
    si.request(1)
    si.get_public_dns_names()
    
    c3.large 2	7	3.75	2 x 16 SSD	$0.105 per Hour
    c4.large 2	8	3.75	EBS Only	$0.11 per Hour
    
    XGBOOST 5   -> ami-d3c5d5e3
    XGBOOST 5.1 -> ami-f38292c3
    XGBOOST 5.2 -> ami-85beaeb5
    XGBOOST 5.3 -> ami-b3627183
    XGBOOST 5.4 -> ami-813a29b1
    XGBOOST 5.5 -> ami-e1617dd1
    XGBOOST 5.6 -> ami-9d1509ad
    XGBOOST 6.0 -> ami-5d98846d
    XGBOOST 6.1 -> ami-57948867
    XGBOOST 6.2 -> ami-c1809cf1
    XGBOOST 6.3 -> ami-1bbda12b
    XGBOOST 6.4 -> ami-eb706fdb
    XGBOOST 6.5 -> ami-afe3fc9f
    """
    def __init__(self, image_id = "ami-afe3fc9f"):
        self.image_id = image_id
        self.conn = boto.ec2.connect_to_region("us-west-2")

    def request(self, num_instances=1, max_price=0.05, subnet='a'):
        if subnet == 'a':
            subnet_id = 'subnet-04299373'
        elif subnet == 'b':
            subnet_id = 'subnet-705a1915'
        else:
            subnet_id = 'subnet-ac9314f5'

        self.reqs = self.conn.request_spot_instances(price=str(max_price),
                                                     image_id=self.image_id,
                                                     count=num_instances,
                                                     type="one-time",
                                                     key_name="panerapig",
                                                     instance_type="c3.large",
                                                     subnet_id=subnet_id)
        for req in self.reqs:
            self.conn.create_tags(req.id, {'purpose': 'xgboost'})

    def cancel(self):
        for req in self.reqs:
            self.conn.cancel_spot_instance_requests(req.id)

    def update_requests(self):
        self.reqs = self.conn.get_all_spot_instance_requests()

    def print_requests(self):
        for req in self.reqs:
            print(req.id)

    def print_public_dns_names(self):
        self.update_requests()
        reservations = self.conn.get_all_reservations(instance_ids = [req.instance_id for req in self.reqs])
        for res in reservations:
            for instance in res.instances:
                print(instance.public_dns_name)
                          

class ClusterTaskQueue():
    def add_task(self, task_string):
        sqs = boto3.resource('sqs')
        queue = sqs.get_queue_by_name(QueueName='cluster_task_queue')
        queue.send_message(MessageBody=task_string)
                