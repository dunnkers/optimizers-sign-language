# Some fantastic DL project
## Preparing the dataset
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
sh util/download_data.sh
```

4. Combine datasets
```shell
python src/combine_datasets.py $(sh util/data_dir.sh)
```

5. Run training
Test using just 32 samples:
```shell
python src/train_model_test.py
```

Do a full training cycle:
```shell
python src/train_model.py -d $(sh util/data_dir.sh)/data.csv
```

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