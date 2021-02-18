import tensorflow as tf
import numpy as np
import json
import argparse

from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Input


####################### Configure parser #######################

parser = argparse.ArgumentParser(description="Train a model on a sign language dataset")

parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=32)
parser.add_argument('-o', '--optimizer', dest='optimizer', default='adam')
parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=10)
parser.add_argument('-n', '--name', dest='model_name', default='my_model')

args = parser.parse_args()


####################### Set some miscellaneous parameters #######################

seed = 42
AUTOTUNE = tf.data.AUTOTUNE


####################### Set hyperparameters #######################

DIMS = (224, 224, 3)
CLASSES = 29
BATCH_SIZE = args.batch_size
OPTIMIZER = args.optimizer
EPOCHS = args.epochs
MODEL_NAME = args.model_name


####################### Load data #######################

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'asl_alphabet_train/asl_alphabet_train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=DIMS[:2],
    shuffle=True,
    seed=seed,
    validation_split=0.1,
    subset='training',
    interpolation="bilinear",
    follow_links=False,
)

ds_train = ds_train.cache().prefetch(AUTOTUNE)

ds_test = tf.keras.preprocessing.image_dataset_from_directory(
    'asl_alphabet_train/asl_alphabet_train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=DIMS[:2],
    shuffle=True,
    seed=seed,
    validation_split=0.1,
    subset='validation',
    interpolation="bilinear",
    follow_links=False,
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
    metrics=['accuracy'])
print(model.summary())


####################### Train model #######################

hist = model.fit(ds_train, validation_data=ds_test, epochs=EPOCHS)


####################### Save output #######################

model.save(MODEL_NAME)

with open(MODEL_NAME+"_history.txt", 'w') as f:
    json.dump(hist.history, f)