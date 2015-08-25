
noglob aws ec2 request-spot-instances --spot-price "0.15" --instance-count 1 --type "one-time" --launch-specification "{\"ImageId\":\"ami-cddccafd\",\"InstanceType\":\"c4.large\",\"Placement\":{\"AvailabilityZone\":\"us-west-2a\"},\"SubnetId\":\"subnet-04299373\",\"KeyName\":\"panerapig\",\"SecurityGroupIds\":[\"sg-fdcdd598\"]}"


