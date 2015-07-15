https://forums.aws.amazon.com/thread.jspa?threadID=168914
s3://support.elasticmapreduce/bootstrap-actions/ami/3.3.1/ridetherocket-workaround.bash
--bootstrap-action "s3://support.elasticmapreduce/bootstrap-actions/ami/3.1.0/tcpPacketLoss.sh"
--bootstrap-action Path=s3://support.elasticmapreduce/bootstrap-actions/ami/3.3.1/ridetherocket-workaround.bash


#!/bin/bash
set -x
# For handling the ride the rocket kernel bug
# https://bugs.launchpad.net/ubuntu/+source/linux/+bug/1317811
sudo ethtool -K eth0 tso off
sudo ethtool -K eth0 sg off
exit 0

