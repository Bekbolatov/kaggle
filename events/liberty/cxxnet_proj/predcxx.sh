#!/bin/bash

MODEL=$1

if [[ "$MODEL" == "" ]]; then
    echo "please specify the model, e.g. 0004"
    exit
fi

cat liberty.conf stub_pred.conf > temp_pred.conf
echo "model_in = ./models/$MODEL.model" >> temp_pred.conf
echo "" >> temp_pred.conf


cxxnet temp_pred.conf

rm temp_pred.conf


