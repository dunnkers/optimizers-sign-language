# Some fantastic DL project
## Preparing the dataset
1. Follow the setup from the [official Kaggle API repo](https://github.com/Kaggle/kaggle-api#api-credentials). Make sure `~/.kaggle/kaggle.json` exists.

2. Create a virtual environment and install the packages:
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Download the data from Kaggle
```shell
sh util/download_data.sh
```

4. Combine datasets
```shell
python src/combine_datasets.py $(sh util/data_dir.sh)
```