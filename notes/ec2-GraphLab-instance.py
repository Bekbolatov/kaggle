import boto.ec2
import boto3

prices = "https://aws.amazon.com/ec2/pricing/"

big = "c3.8xlarge"
bigger = "r3.8xlarge"


a = 'subnet-04299373'
b = 'subnet-705a1915'
c = 'subnet-ac9314f5'

volume = 'vol-24e7f9ea'

image = 'ami-d3c3dae3'  # GraphLab-1.1
older_image = 'ami-45afb075' # GRaphLab 1.1   //ami-bd6e718d'(GraphLab 1.0)

conn = boto.ec2.connect_to_region("us-west-2")
conn.request_spot_instances(price='4.50', 
        image_id=image,
        count=1, 
        type="one-time", 
        key_name="panerapig",
        instance_type=bigger,
        subnet_id=a)

