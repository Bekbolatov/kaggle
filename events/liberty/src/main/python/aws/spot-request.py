import boto.ec2



class SpotInstances:

    def request(self, num_instances=1, max_price=0.05):
        conn = boto.ec2.connect_to_region("us-west-2")
        self.reqs = conn.request_spot_instances(price=str(max_price), image_id="ami-ff8395cf",count=num_instances,type="one-time", key_name="panerapig",instance_type="c4.large", subnet_id="subnet-04299373")
        for req in self.reqs:
            conn.create_tags(req.id, {'purpose': 'xgboost'})

    def cancel(self):
        for req in self.reqs:
            conn.cancel_spot_instance_requests(req.id)

    def get_requests(self):
        reqs = conn.get_all_spot_instance_requests()
        return reqs

    def print_requests(reqs):
        for req in reqs:
            print(req.id)


