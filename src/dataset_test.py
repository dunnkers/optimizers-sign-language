import pandas as pd
from dataset import getdataset

# data = pd.read_csv('../datasets/data.csv')
data = pd.DataFrame({
    'file_path': ['../test_data/O.jpg', '../test_data/D.jpeg'],
    'class': ['O', 'D']
})
dataset = getdataset(data)
print(dataset)