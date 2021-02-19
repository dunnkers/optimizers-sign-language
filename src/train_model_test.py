import unittest
from train_model import train_model
from combine_datasets import combine_datasets
from argparse import Namespace

class TestTrainModel(unittest.TestCase):
    def test_train_model(self):
        data = combine_datasets('./test_data', class_encoding='file')

        args = Namespace()
        args.batch_size = 32
        args.optimizer = 'adam'
        args.epochs = 1
        args.steps_per_epoch = None
        args.validation_steps = None

        hist, model = train_model(data, args)
        print(hist)

if __name__ == '__main__':
    unittest.main()
