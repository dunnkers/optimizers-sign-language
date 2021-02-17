#!/bin/bash
source venv/bin/activate
data_dir=$(src/data-dir.sh)
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

# # merged dataset target folder
# dataset=$data_dir/sign-language
# mkdir -p $dataset

# # merge datasets
# (find $data_dir -regex '.*\asl_alphabet_train/[A-Z]\/.*\.jpg';\
#  find $data_dir -regex '.*\asl_dataset/[a-z]\/.*\.jpeg';\
#  find $data_dir -regex '.*\Gesture Image Data/[A-Z]\/.*\.jpg';) | cat\
#     |while read fpath; do
#   class=$(echo $fpath | sed -E 's/(.*\/)([A-Z])(\/.*\.jp(e|.*)g)/\2/g')
#   class=$(echo $class | tr a-z A-Z)
#   fname=$(echo $fpath | sed -E 's/(.*\/)([A-Z])\/(.*\.jp(e|.*)g)/\3/g')
#   echo "$fpath is file $fname is of class $class"

#   echo "Copying ${fpath} to $dataset/$class/$fname..."
#   mkdir -p $dataset/$class/
#   cp -R "$fpath" "$dataset/$class/$fname"
# done