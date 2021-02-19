import tensorflow as tf
import pandas as pd
import json
import argparse
import os

from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Input
from keras.metrics import top_k_categorical_accuracy

from dataset import getdataset
from time_history import TimeHistory


####################### Configure parser #######################

parser = argparse.ArgumentParser(description="Train a model on a sign language dataset")

parser.add_argument('-p', '--path', dest='data_path', required=True)
parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=32)
parser.add_argument('-o', '--optimizer', dest='optimizer', default='adam')
parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=10)
parser.add_argument('-s', '--steps-per-epoch', dest='steps_per_epoch', type=int)
parser.add_argument('-v', '--validation-steps', dest='validation_steps', type=int)
parser.add_argument('-d', '--output_dir', dest='output_dir', default='models')
parser.add_argument('-n', '--name', dest='model_name', default='my_model')

args = parser.parse_args()


####################### Set some miscellaneous parameters #######################

seed = 42
AUTOTUNE = tf.data.AUTOTUNE
DATA_PATH = args.data_path
OUTPUT_DIR = args.output_dir


####################### Set hyperparameters #######################

DIMS = (224, 224, 3)
CLASSES = 26
BATCH_SIZE = args.batch_size
OPTIMIZER = args.optimizer
EPOCHS = args.epochs
STEPS_PER_EPOCH = args.steps_per_epoch
VALIDATION_STEPS = args.validation_steps
MODEL_NAME = args.model_name


####################### Load data #######################

data_paths = pd.read_csv(DATA_PATH)

ds_train = getdataset(
    data_paths,
    batch_size=BATCH_SIZE,
    input_shape=DIMS[:2],
    seed=seed,
    validation_split=0.1,
    subset='training'
)

ds_train = ds_train.cache().prefetch(AUTOTUNE)

ds_test = getdataset(
    data_paths,
    batch_size=BATCH_SIZE,
    input_shape=DIMS[:2],
    seed=seed,
    validation_split=0.1,
    subset='validation'
)

ds_test = ds_test.cache().prefetch(AUTOTUNE)


####################### Configure model #######################

i = Input(DIMS)
x = tf.keras.applications.mobilenet_v3.preprocess_input(i)
x = MobileNetV3Small(input_tensor=x, classes=CLASSES, weights=None)(x)

model = tf.keras.Model(inputs=i, outputs=x)

model.compile(
    optimizer=OPTIMIZER, 
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
    metrics=['accuracy',
             'categorical_accuracy',
             top_k_categorical_accuracy])
print(model.summary())

time_callback = TimeHistory()


####################### Train model #######################

hist = model.fit(ds_train, 
                 validation_data=ds_test, 
                 epochs=EPOCHS,
                 steps_per_epoch=STEPS_PER_EPOCH,
                 validation_steps=VALIDATION_STEPS,
                 callbacks=[time_callback])


####################### Save output #######################

out_dir = os.path.join(OUTPUT_DIR, MODEL_NAME)
model.save(out_dir)

# Add epoch times to history
hist.history['epoch_time'] = time_callback.times

out_file = os.path.join(OUTPUT_DIR, 'history.json')
with open(out_file, 'w') as f:
    json.dump(hist.history, f)