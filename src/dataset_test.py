import unittest
import string
from dataset import getdataset
from combine_datasets import combine_datasets

class TestSum(unittest.TestCase):
    def setUp(self):
        self.data = combine_datasets('./test_data', class_encoding='file')
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
