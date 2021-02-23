import unittest
import string
from dataset import getdataset
from combine_datasets import combine_datasets
import numpy as np
import pandas as pd

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.data = combine_datasets('./test_data', class_encoding='file')

    def test_classnames(self):
        dataset = getdataset(self.data)
        self.assertTrue(all(dataset.class_names == list(string.ascii_uppercase)))

    def test_training_split(self):
        dataset = getdataset(self.data, seed=343, validation_split=0.25,
            subset='training')
        imgs, labels = next(iter(dataset.take(1)))
        self.assertTrue(len(imgs) == 24)

    def test_validation_split(self):
        dataset = getdataset(self.data, seed=343, validation_split=0.25,
            subset='validation')
        imgs, labels = next(iter(dataset.take(1)))
        self.assertTrue(len(imgs) == 8)

    def test_label_encoding(self):
        dataset = getdataset(self.data)
        ims, labels = next(iter(dataset.take(1)))
        self.assertEqual(np.shape(labels.shape)[0], 2)
        self.assertEqual(labels.shape[0], 32)
        self.assertEqual(labels.shape[1], 26)

    def test_cv_label_encoding(self):
        dataset = getdataset(self.data, seed=343, validation_split=0.25,
            subset='training')
        ims, labels = next(iter(dataset.take(1)))
        self.assertEqual(np.shape(labels.shape)[0], 2)
        self.assertEqual(labels.shape[0], 24)
        self.assertEqual(labels.shape[1], 26)

    def test_correct_labels(self):
        dataset = getdataset(self.data, shuffle=None)
        ims, labels = next(iter(dataset.take(1)))
        a = [np.where(label)[0][0] for label in labels] # one-hot to int vector
        b, _ = pd.factorize(self.data['class'])         # integer vector
        self.assertTrue(np.array_equal(a, b))
    
    def test_shuffled(self):
        dataset = getdataset(self.data, shuffle=True)
        ims, labels = next(iter(dataset.take(1)))
        a = [np.where(label)[0][0] for label in labels] # one-hot to int vector
        b, _ = pd.factorize(self.data['class'])         # integer vector
        self.assertFalse(np.array_equal(a, b))
    
    def test_shuffled_before_split(self):
        """The dataset should be shuffled *before* training/testing split.
        Otherwise, the validation set might only contain certain classes, i.e.
        classes that the training set does not even contain."""
        dataset = getdataset(self.data, seed=343, validation_split=0.25,
            subset='validation')
        ims, labels = next(iter(dataset.take(1)))
        a = [np.where(label)[0][0] for label in labels] # one-hot to int vector
        b, _ = pd.factorize(self.data['class'])         # integer vector
        last_8 = b[-8:]
        a.sort()
        last_8.sort()
        self.assertFalse(np.array_equal(a, last_8))

if __name__ == '__main__':
    unittest.main()
