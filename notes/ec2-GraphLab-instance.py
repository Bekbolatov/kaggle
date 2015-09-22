import boto.ec2
import boto3

prices = "https://aws.amazon.com/ec2/pricing/"

big = "c3.8xlarge"
bigger = "r3.8xlarge"


a = 'subnet-04299373'
b = 'subnet-705a1915'
c = 'subnet-ac9314f5'

volume = 'vol-24e7f9ea'


conn = boto.ec2.connect_to_region("us-west-2")
conn.request_spot_instances(price='1.50', 
        image_id='ami-45afb075', # GraphLab 1.1      #'ami-bd6e718d'(GraphLab 1.0)
        count=1, 
        type="one-time", 
        key_name="panerapig",
        instance_type=bigger,
        subnet_id=a)

