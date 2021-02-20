import os
from tensorflow.keras.callbacks import Callback
import pandas as pd
import time

class BatchLoss(Callback):
    """Saves a loss history to a `.csv` file on each batch end."""
    def __init__(self, output_file):
        self.output_file = output_file

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch

    def on_batch_begin(self, batch, logs={}):
        self.batch_time_start = time.time()

    def on_batch_end(self, batch, logs={}):
        data = pd.DataFrame.from_records([logs])
        data['batch'] = batch
        data['epoch'] = self.epoch
        data['batch_time'] = time.time() - self.batch_time_start
        if not os.path.isfile(self.output_file):
            data.to_csv(self.output_file, index=False)
        else:
            data.to_csv(self.output_file, mode='a', index=False, header=False)