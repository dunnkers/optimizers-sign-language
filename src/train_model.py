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
                callbacks=[],
                weights=None):
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

    print(f'Using weights: {weights}')
    if weights == None: # learn from scratch
        i = Input(dims)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(i)
        x = MobileNetV2(classes=classes, weights=None)(x)
        model = tf.keras.Model(inputs=[i], outputs=[x])
    else:               # apply transfer learning
        base_model = MobileNetV2(
            include_top=False,
            weights=weights,
            input_shape=dims)
        base_model.trainable = False
        
        # layers
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        dense_layer = tf.keras.layers.Dense(1024, activation='relu')
        prediction_layer = tf.keras.layers.Dense(26, activation='softmax')

        # construct pipeline
        inputs = tf.keras.Input(shape=dims)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = global_average_layer(x)
        x = dense_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)


    model.compile(
        optimizer=args.optimizer, 
        loss=tf.keras.losses.CategoricalCrossentropy(), 
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
    parser.add_argument('-l', '--inv-learning-rate', dest='inv_learning_rate', type=int, default=1000)
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=10)
    parser.add_argument('-s', '--steps-per-epoch', dest='steps_per_epoch', type=int)
    parser.add_argument('-v', '--validation-steps', dest='validation_steps', type=int)
    parser.add_argument('-d', '--output_dir', dest='output_dir', default='models')
    parser.add_argument('-n', '--name', dest='model_name', default='my_model')
    parser.add_argument('-t', '--take-samples', dest='samples', type=int, default=None)
    parser.add_argument('-w', '--weights', dest='weights', default=None)
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
    if args.optimizer._name.lower() == 'sgd':
        optim_config['momentum'] = 0.9
        optim_config['nesterov'] = True
    if args.optimizer._name.lower() == 'rmsprop':
        optim_config['momentum'] = 0.9
    if args.optimizer._name.lower() == 'ftrl':
        optim_config['l2_regularization_strength'] = 1.0
    optim_config['learning_rate'] = 1 / args.inv_learning_rate
    df_config = pd.DataFrame.from_records([ optim_config ])
    df_config.to_csv(os.path.join(out_dir, 'optimizer.csv'), index=False)
    print(f'Optimizer config:')
    print(df_config)
    # Adapt new optimizer config
    args.optimizer = args.optimizer.from_config(optim_config)

    ####################### Callbacks  #######################
    
    batch_loss = BatchLoss(os.path.join(out_dir, 'batch_loss.csv'))
    epoch_loss = EpochLoss(os.path.join(out_dir, 'epoch_loss.csv'))
    model_file = os.path.join(out_dir, 'checkpoints')
    model_checkpoint = ModelCheckpoint(model_file,
                                       verbose=1,
                                       save_best_only=True)


    ####################### Train model #######################

    data_paths = pd.read_csv(args.data_path)
    if 'dataset' in data_paths and args.samples != None:
        n_datasets = len(data_paths['dataset'].unique())
        print(f'Taking {args.samples} samples from {n_datasets} datasets.')
        counts = data_paths.groupby('dataset')\
           .agg({'filepath': 'count'})\
           .rename(columns={'filepath': 'samples'})
        df = data_paths.join(counts, on='dataset')
        df['sample_probability'] = df['samples'].unique().sum() / df['samples']
        data_paths = df.sample(n=args.samples, weights='sample_probability')
    hist, model = train_model(data_paths, args,
        callbacks=[epoch_loss, model_checkpoint], weights=args.weights)

    ####################### Save output #######################

    model.save(out_dir)