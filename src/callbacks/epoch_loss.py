import time
from tensorflow.keras.callbacks import CSVLogger

class EpochLoss(CSVLogger):
    def on_train_begin(self, logs={}):
        self.times = []
        super(EpochLoss, self).on_train_begin()

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()
        super(EpochLoss, self).on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs={}):
        logs['epoch_time'] = time.time() - self.epoch_time_start
        super(EpochLoss, self).on_epoch_end(epoch, logs)