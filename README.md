# Some fantastic DL project
## Preparing the dataset
1. Follow the setup from the [official Kaggle API repo](https://github.com/Kaggle/kaggle-api#api-credentials). Make sure `~/.kaggle/kaggle.json` exists.

2. Create a virtual environment and install the packages:
```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Download the data from Kaggle
```shell
sh src/download-data.sh
```

4. (Optional) Combine and filter the data
Run the `src/data-exploration.ipynb` Notebook in its entirety. It will produce a file at `dataset/data.csv`, describing the dataset.

5. Use the dataset 💠 See `src/data-set.py` for an example of constructing a Keras `Dataset` object.