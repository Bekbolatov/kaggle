import boto.ec2



class SpotInstances:
    """
    si = SpotInstances()
    si.request(1)
    si.get_public_dns_names()
    """
    def __init__(self, image_id = "ami-d3c5d5e3"):
        self.image_id = image_id

    def request(self, num_instances=1, max_price=0.05):
        self.conn = boto.ec2.connect_to_region("us-west-2")
        self.reqs = self.conn.request_spot_instances(price=str(max_price),
                                                     image_id=self.image_id,
                                                     count=num_instances,
                                                     type="one-time",
                                                     key_name="panerapig",
                                                     instance_type="c4.large",
                                                     subnet_id="subnet-04299373")
        for req in self.reqs:
            self.conn.create_tags(req.id, {'purpose': 'xgboost'})

    def cancel(self):
        for req in self.reqs:
            self.conn.cancel_spot_instance_requests(req.id)

    def get_requests(self):
        self.reqs = self.conn.get_all_spot_instance_requests()
        return self.reqs

    def print_requests(self):
        for req in self.reqs:
            print(req.id)

