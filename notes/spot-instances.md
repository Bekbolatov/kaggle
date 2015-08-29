

noglob aws ec2 request-spot-instances --spot-price "0.15" --instance-count 40 --type "one-time" --launch-specification "{\"ImageId\":\"ami-ff8395cf\",\"InstanceType\":\"c4.large\",\"Placement\":{\"AvailabilityZone\":\"us-west-2a\"},\"SubnetId\":\"subnet-04299373\",\"KeyName\":\"panerapig\",\"SecurityGroupIds\":[\"sg-fdcdd598\"]}"

noglob aws ec2 request-spot-instances --spot-price "0.12" --instance-count 65 --type "one-time" --launch-specification "{\"ImageId\":\"ami-5ff0e66f\",\"InstanceType\":\"c4.large\",\"Placement\":{\"AvailabilityZone\":\"us-west-2b\"},\"SubnetId\":\"subnet-705a1915\",\"KeyName\":\"panerapig\",\"SecurityGroupIds\":[\"sg-fdcdd598\"]}"

aws ec2 describe-instances --filters Name=tag:purpose,Values=xgboost | jq '.Reservations[].Instances[].PublicIpAddress'
aws ec2 describe-instances --filters  Name=image-id,Values=ami-cddccafd | jq '.Reservations[].Instances[].PublicIpAddress'

aws ec2 describe-instances --filters  Name=image-id,Values=ami-ff8395cf | jq '.Reservations[].Instances[].PublicDnsName'




noglob aws ec2 request-spot-instances --spot-price "0.04" --instance-count 1 --type "one-time" --launch-specification "{\"ImageId\":\"ami-ff8395cf\",\"InstanceType\":\"c4.large\",\"Placement\":{\"AvailabilityZone\":\"us-west-2a\"},\"SubnetId\":\"subnet-04299373\",\"KeyName\":\"panerapig\",\"SecurityGroupIds\":[\"sg-fdcdd598\"]}"
