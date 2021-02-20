from enum import Enum
from tensorflow.python.keras.optimizers import get

class OptimizerEncoding(Enum):
    """ See `tensorflow.python.keras.optimizers` """
    adadelta = 0
    adagrad  = 1
    adam     = 2
    adamax   = 3
    nadam    = 4
    rmsprop  = 5
    sgd      = 6
    ftr      = 7

def get_optimizer(optimizer):
    if optimizer.isdigit(): # choose by int encoding
        optimizer = OptimizerEncoding(int(optimizer)).name
    return get(optimizer)