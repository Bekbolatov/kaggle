import boto.ec2
import boto3

prices = "https://aws.amazon.com/ec2/pricing/"

big = "c3.8xlarge"
bigger = "r3.8xlarge"


a = 'subnet-04299373'
b = 'subnet-705a1915'
c = 'subnet-ac9314f5'

volume = 'vol-24e7f9ea'

new_image = 'ami-9615f0a5' # GraphLab-1.2
image = 'ami-d3c3dae3'  # GraphLab-1.1
older_image = 'ami-45afb075' # GRaphLab 1.1   //ami-bd6e718d'(GraphLab 1.0)

#/dev/xvda=snap-56188a0b:16:true:gp2
#https://aloysius.wordpress.com/2014/07/19/auto-tag-ec2-spot-instances-and-volumes-with-boto/
#bdm_config = {
#    "/dev/sdf" : "vol-24e7f9ea: : false:gp2" 
#}
#bdm = boto.ec2.blockdevicemapping.BlockDeviceMapping()
#for name, bd in bdm_config.iteritems():
#    bdm[name] = boto.ec2.blockdevicemapping.BlockDeviceType(**bd)




conn = boto.ec2.connect_to_region("us-west-2")
conn.request_spot_instances(price='4.50', 
        image_id=new_image,
        count=1, 
        type="one-time", 
        key_name="panerapig",
        instance_type=bigger,
        subnet_id=a)

