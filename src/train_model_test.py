import unittest
from train_model import train_model
from combine_datasets import combine_datasets
from argparse import Namespace
import tensorflow as tf
import tempfile
import numpy as np

class TestTrainModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = Namespace()
        args.batch_size = 32
        args.optimizer = 'adam'
        args.inv_learning_rate = 100 # inverse: so converted to 0.001
        args.epochs = 1
        args.steps_per_epoch = None
        args.validation_steps = None
        cls.args = args
        cls.data = combine_datasets('./test_data', class_encoding='file')
        _, model = train_model(cls.data, cls.args)
        cls.model = model
        cls.x = tf.random.uniform((32, 224, 224, 3)) # a random batch

    def test_one_epoch(self):
        """Weights should have changed; i.e. no uniform prediction"""
        self.assertFalse(np.allclose(
            self.model.predict(self.x)[0], # prediction
            np.ones(26)/26)                # uniform probability vector
        )

    def test_probability_vector(self):
        self.assertTrue(np.isclose(np.sum(self.model.predict(self.x)[0]), 1))

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as dirpath:
            self.model.save(dirpath)
            loaded_model = tf.keras.models.load_model(dirpath)
            real   =   self.model.predict(self.x)
            loaded = loaded_model.predict(self.x)
            self.assertTrue(np.allclose(real, loaded))

    def test_imagenet(self):
        """ Predictions should be non-uniform right from the start - without
        even running 1 epoch."""
        self.args.epochs = 0
        _, model = train_model(self.data, self.args, weights='imagenet')
        self.assertFalse(np.allclose(
            model.predict(self.x)[0],      # prediction
            np.ones(26)/26)                # uniform probability vector
        )

if __name__ == '__main__':
    unittest.main()
