#!/bin/bash
repo_dir=$(git rev-parse --show-toplevel)
data_dir="$repo_dir/datasets"
if [[ $HOSTNAME == *"hpc.rug.nl"* ]]; then
  data_dir="/data/$USER"
fi
echo $data_dir