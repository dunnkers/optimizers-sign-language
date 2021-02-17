#!/bin/bash
root_dir=$(src/root-dir.sh)
data_dir="$root_dir/datasets"
if [[ $HOSTNAME == *"peregrine"* ]]; then
  data_dir="/data/$USER/"
fi
echo $data_dir