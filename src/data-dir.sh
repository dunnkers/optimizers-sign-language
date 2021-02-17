#!/bin/bash
data_dir="../datasets"
if [[ $HOSTNAME == *"peregrine"* ]]; then
  data_dir="/data/$USER/"
fi
echo $data_dir