#!/bin/bash

PIPE=/home/ec2-user/input_queue


while read line
do
    if [ ! -z "$line" ]; then
        echo $line
        tokens=( $line )
        TASK_ID=${tokens[0]}
        ARGS="${tokens[1]}"

        export DATA_LOCATION=/home/ec2-user/data/liberty
        export OUTPUT_LOCATION_DIR=/home/ec2-user
        export OUTPUT_LOCATION=${OUTPUT_LOCATION_DIR}/TASK_${TASK_ID}

        rm -rf $OUTPUT_LOCATION
        mkdir $OUTPUT_LOCATION

        cd /home/ec2-user/repos/bekbolatov/kaggle/events/liberty/src/main/python
        RESULT=$(python xgboost_liberty_stack.py "$ARGS" 1>$OUTPUT_LOCATION/out.log 2>$OUTPUT_LOCATION/error.log)

        echo "$ARGS" > ${OUTPUT_LOCATION}/args.txt
        touch ${OUTPUT_LOCATION}/task_done
    fi
done <$PIPE

