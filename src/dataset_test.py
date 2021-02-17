import unittest
import pandas as pd
from dataset import getdataset
import string
import tensorflow as tf
from keras.metrics import top_k_categorical_accuracy
from keras.optimizers import Adam
import subprocess as sp

class TestSum(unittest.TestCase):
    def setUp(self):
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
    
    def test_training(self):
        root_dir = sp.getoutput('./src/root-dir.sh')
        data = pd.read_csv('./test_data/index.csv')
        data['filepath'] = f'{root_dir}/' + data['filepath']
        dataset = getdataset(data)

        model = tf.keras.applications.MobileNetV3Small(weights=None, classes=26)
        model.compile(  optimizer=Adam(lr=1e-4), 
                        loss='categorical_crossentropy',
                        metrics=[
                            'categorical_accuracy',
                            top_k_categorical_accuracy
                        ])
        model.fit(x=dataset)

if __name__ == '__main__':
    unittest.main()
