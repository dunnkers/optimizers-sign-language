import tensorflow as tf
import pandas as pd
import argparse
import os

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.callbacks import ModelCheckpoint

from dataset import getdataset
from callbacks.batch_loss import BatchLoss
from callbacks.epoch_loss import EpochLoss
from get_optimizer import get_optimizer

def train_model(data_paths,
                args,
                seed=42,
                classes=26,
                dims=(224, 224, 3),
                callbacks=[]):
    ####################### Load data #######################

    ds_train = getdataset(
        data_paths,
        batch_size=args.batch_size,
        input_shape=dims[:2],
        seed=seed,
        validation_split=0.1,
        subset='training'
    )
    ds_train = ds_train.cache().prefetch(2)

    ds_test = getdataset(
        data_paths,
        batch_size=args.batch_size,
        input_shape=dims[:2],
        seed=seed,
        validation_split=0.1,
        subset='validation'
    )
    ds_test = ds_test.cache().prefetch(2)


    ####################### Configure model #######################

    i = Input(dims)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(i)
    x = MobileNetV2(input_tensor=x, classes=classes, weights=None)(x)

    model = tf.keras.Model(inputs=i, outputs=x)

    model.compile(
        optimizer=args.optimizer, 
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
        metrics=['accuracy',
                'categorical_accuracy',
                top_k_categorical_accuracy])
    print(model.summary())


    ####################### Train model #######################

    hist = model.fit(ds_train, 
                    validation_data=ds_test, 
                    epochs=args.epochs,
                    steps_per_epoch=args.steps_per_epoch,
                    validation_steps=args.validation_steps,
                    callbacks=callbacks)

    return hist, model


if __name__ == '__main__':
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

    ####################### Out directory  #######################

    out_dir = os.path.join(args.output_dir, args.model_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ####################### Optimizers  #######################
    args.optimizer = get_optimizer(args.optimizer)
    print(f'Using optimizer: {args.optimizer._name}')
    # Determine and print config
    optim_config = args.optimizer.get_config()
    df_config = pd.DataFrame.from_records([ args.optimizer.get_config() ])
    df_config.to_csv(os.path.join(out_dir, 'optimizer.csv'), index=False)
    print(f'Optimizer config: \n{df_config}')
    # Adapt new optimizer config
    args.optimizer = args.optimizer.from_config(optim_config)

    ####################### Callbacks  #######################
    
    batch_loss = BatchLoss(os.path.join(out_dir, 'batch_loss.csv'))
    epoch_loss = EpochLoss(os.path.join(out_dir, 'epoch_loss.csv'))
    model_file = os.path.join(out_dir, 'checkpoints', 
        r'epoch={epoch},val_loss={val_loss:.2f}.h5')
    model_checkpoint = ModelCheckpoint(model_file, verbose=1)


    ####################### Train model #######################

    data_paths = pd.read_csv(args.data_path)
    hist, model = train_model(data_paths, args,
        callbacks=[model_checkpoint, batch_loss, epoch_loss])

    ####################### Save output #######################

    model.save(out_dir)