
noglob aws ec2 request-spot-instances --spot-price "0.15" --instance-count 1 --type "one-time" --launch-specification "{\"ImageId\":\"ami-cddccafd\",\"InstanceType\":\"c4.large\",\"Placement\":{\"AvailabilityZone\":\"us-west-2a\"},\"SubnetId\":\"subnet-04299373\",\"KeyName\":\"panerapig\",\"SecurityGroupIds\":[\"sg-fdcdd598\"]}"


=========   WITHOUT SETTING SECURITY GROUP ID   =========

noglob aws ec2 request-spot-instances --spot-price "0.15" --instance-count 1 --type "one-time" --launch-specification "{\"ImageId\":\"ami-cddccafd\",\"InstanceType\":\"c4.large\",\"Placement\":{\"AvailabilityZone\":\"us-west-2a\"},\"SubnetId\":\"subnet-04299373\",\"KeyName\":\"panerapig\"}"



aws ec2 describe-instances --filters Name=tag:purpose,Values=xgboost | jq '.Reservations[].Instances[].PublicIpAddress'

aws ec2 describe-instances --filters  Name=image-id,Values=ami-cddccafd | jq '.Reservations[].Instances[].PublicIpAddress'
aws ec2 describe-instances --filters  Name=image-id,Values=ami-cddccafd | jq '.Reservations[].Instances[].PublicDnsName'
