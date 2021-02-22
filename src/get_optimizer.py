from enum import Enum
from tensorflow.python.keras.optimizers import get
from adabelief_tf import AdaBeliefOptimizer

class OptimizerEncoding(Enum):
    """ See `tensorflow.python.keras.optimizers` """
    adadelta  = 0
    adagrad   = 1
    adam      = 2
    adamax    = 3
    nadam     = 4
    rmsprop   = 5
    sgd       = 6
    ftrl      = 7
    adabelief = 8

def get_optimizer(optimizer):
    if optimizer.isdigit(): # choose by int encoding
        optimizer = OptimizerEncoding(int(optimizer)).name
    if optimizer == 'adabelief':
        return AdaBeliefOptimizer(learning_rate=1e-3, epsilon=1e-8,
            rectify=False, weight_decay=1e-2)
    else:
        return get(optimizer)