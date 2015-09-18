import boto.ec2
import boto3

conn = boto.ec2.connect_to_region("us-west-2")
conn.request_spot_instances(price='0.50', 
        image_id='ami-45afb075', # GraphLab 1.1      #'ami-bd6e718d'(GraphLab 1.0)
        count=1, 
        type="one-time", 
        key_name="panerapig",
        instance_type="r3.2xlarge",
        subnet_id='subnet-04299373')

