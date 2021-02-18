import unittest
import pandas as pd
from dataset import getdataset
from combine_datasets import combine_datasets
import string

class TestSum(unittest.TestCase):
    def setUp(self):
        # data_dir = sp.getoutput('../util/data_dir.sh')
        # data = pd.read_csv('./test_data/index.csv')
        self.data = pd.read_csv('./test_data/index.csv')
        self.dataset = getdataset(self.data)

    def test_classnames(self):
        self.assertTrue(all(self.dataset.class_names == list(string.ascii_uppercase)))

    def test_training_split(self):
        dataset = getdataset(self.data, seed=343, validation_split=0.25, subset='training')
        imgs, labels = next(iter(dataset.take(1)))
        self.assertTrue(len(imgs) == 24)

    def test_validation_split(self):
        dataset = getdataset(self.data, seed=343, validation_split=0.25, subset='validation')
        imgs, labels = next(iter(dataset.take(1)))
        self.assertTrue(len(imgs) == 8)

if __name__ == '__main__':
    unittest.main()
