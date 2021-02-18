#!/bin/bash
source venv/bin/activate
data_dir=$(util/data-dir.sh)
echo "Using dataset directory: $data_dir"

# grassknoted/asl-alphabet
kaggle datasets download -p $data_dir grassknoted/asl-alphabet
unzip -n $data_dir/asl-alphabet.zip -d $data_dir/

# ayuraj/asl-dataset
kaggle datasets download -p $data_dir ayuraj/asl-dataset
unzip -n $data_dir/asl-dataset.zip -d $data_dir/

# ahmedkhanak1995/sign-language-gesture-images-dataset
kaggle datasets download -p $data_dir ahmedkhanak1995/sign-language-gesture-images-dataset
unzip -n $data_dir/sign-language-gesture-images-dataset.zip -d $data_dir/