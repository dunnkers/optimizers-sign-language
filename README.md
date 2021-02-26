# Benchmarking Optimizers for Sign Language detection
Deep Learning (20/21) `WMAI017-05.2020-2021.2A`

Uses [data](https://www.kaggle.com/grassknoted/asl-alphabet) describing the ASL alphabet and tries to classify the images correctly using an adapted custom Neural Network, built with TensorFlow/Keras. Runs the model fitting multiple times for various optimizers, such that we can compare various optimizers against each other.

## Usage
1. Follow the setup from the [official Kaggle API repo](https://github.com/Kaggle/kaggle-api#api-credentials). Make sure `~/.kaggle/kaggle.json` exists.

2. Create a virtual environment and install the packages:
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Will install all required packages.

3. Download the data from Kaggle
```shell
sh util/data_download.sh <directory_to_store_data>
```

4. (optional) Combine datasets
```shell
python src/combine_datasets.py <directory_to_store_data>
```

5. Run training
Test using just 32 samples:
```shell
python src/train_model_test.py
```

Do a full training cycle:
```shell
python src/train_model.py -d <directory_to_store_data>/<dataset_to_use>
```

Directory `<dataset_to_use>` must have subdirectories containing the names of the designated classes.

## Peregrine
Follow the instructions above, will work for Peregrine just as well. Submit a job using:

```shell
sbatch util/peregrine.sh
```

Download the results using:
```shell
rsync -aP $PEREGRINE_USERNAME@peregrine.hpc.rug.nl:~/deep-learning/logs ./
rsync -aP $PEREGRINE_USERNAME@peregrine.hpc.rug.nl:~/deep-learning/models ./
```

## Demo
Run the demo locally by running:

```shell
cd demo
yarn
yarn start
```

A browser tab should automatically be opened with the website ✨

## About
By [Jeroen Overschie](https://dunnkers.com) and Loran Knol.