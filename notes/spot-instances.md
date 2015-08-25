

noglob aws ec2 request-spot-instances --spot-price "0.15" --instance-count 30 --type "one-time" --launch-specification "{\"ImageId\":\"ami-ddf7e1ed\",\"InstanceType\":\"c4.large\",\"Placement\":{\"AvailabilityZone\":\"us-west-2a\"},\"SubnetId\":\"subnet-04299373\",\"KeyName\":\"panerapig\",\"SecurityGroupIds\":[\"sg-fdcdd598\"]}"


aws ec2 describe-instances --filters Name=tag:purpose,Values=xgboost | jq '.Reservations[].Instances[].PublicIpAddress'
aws ec2 describe-instances --filters  Name=image-id,Values=ami-cddccafd | jq '.Reservations[].Instances[].PublicIpAddress'

aws ec2 describe-instances --filters  Name=image-id,Values=ami-ddf7e1ed | jq '.Reservations[].Instances[].PublicDnsName'
