import unittest
from get_optimizer import get_optimizer
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.optimizer_v2 import adadelta as adadelta_v2

class GetOptimizerTest(unittest.TestCase):
    def test_digit_input(self):
        optimizer = get_optimizer('0')
        self.assertIsInstance(optimizer, optimizer_v2.OptimizerV2)
        self.assertIsInstance(optimizer, adadelta_v2.Adadelta)

if __name__ == '__main__':
    unittest.main()
