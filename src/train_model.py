import pandas as pd
import argparse
import os

from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from numpy.random import seed
from tensorflow import set_random_seed

from keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Nadam, RMSprop, SGD
from optimizers.AdaBeliefOptimizer import AdaBeliefOptimizer
from optimizers.RAdamOptimizer import RAdamOptimizer
from optimizers.Yogi import Yogi

from callbacks.epoch_loss import EpochLoss

def train_model(data_path,
                args,
                classes=29,
                dims=(64, 64, 3),
                callbacks=[]):
    ####################### Load data #######################
    data_augmentor = ImageDataGenerator(samplewise_center=True, 
                                        samplewise_std_normalization=True, 
                                        validation_split=0.1)
    train_generator = data_augmentor.flow_from_directory(data_path,
        target_size=dims[:2], batch_size=args.batch_size, shuffle=True,
        subset="training")
    val_generator = data_augmentor.flow_from_directory(data_path,
        target_size=dims[:2], batch_size=args.batch_size,
        subset="validation")


    ####################### Configure model #######################

    my_model = Sequential()
    my_model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=target_dims))
    my_model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
    my_model.add(Dropout(0.5))
    my_model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))
    my_model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))
    my_model.add(Dropout(0.5))
    my_model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu'))
    my_model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu'))
    my_model.add(Flatten())
    my_model.add(Dropout(0.5))
    my_model.add(Dense(512, activation='relu'))
    my_model.add(Dense(classes, activation='softmax'))


    my_model.compile(
        optimizer=args.optimizer, 
        loss='categorical_crossentropy', 
        metrics=['accuracy'])
    print(my_model.summary())


    ####################### Train model #######################

    my_model.fit_generator(train_generator, epochs=args.epochs,
        validation_data=val_generator, callbacks=callbacks)


    return my_model


if __name__ == '__main__':
    ####################### Configure parser #######################

    parser = argparse.ArgumentParser(description="Train a model on a sign language dataset")
    parser.add_argument('-p', '--path', dest='data_path', required=True)
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=32)
    parser.add_argument('-o', '--optimizer', dest='optimizer', default='adam')
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=10)
    parser.add_argument('-d', '--output_dir', dest='output_dir', default='models')
    parser.add_argument('-n', '--name', dest='model_name', default='my_model')
    args = parser.parse_args()

    ####################### Out directory  #######################

    out_dir = os.path.join(args.output_dir, args.model_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ####################### Optimizers  #######################

    lr = 0.001
    optimizers = [
        ('Yogi', Yogi(lr=lr)),
        ('AdaBelief', AdaBeliefOptimizer(learning_rate=lr)),
        ('RAdam', RAdamOptimizer(learning_rate=lr)),
        ('Adadelta', Adadelta(lr=lr)),
        ('Adagrad', Adagrad(lr=lr)),
        ('Adam', Adam(lr=lr)),
        ('Adamax', Adamax(lr=lr)),
        ('Nadam', Nadam(lr=lr)),
        ('RMSprop', RMSprop(lr=lr)),
        ('SGD', SGD(lr=lr, momentum=0.9, nesterov=True))
    ]
    name = None
    optim = None
    for optimizer in optimizers:
        n, obj = optimizer
        if n == args.optimizer.tolower():
            name = n
            optim = obj
    optim_path = os.path.join(out_dir, 'optimizer.json')
    try:
        conf = getattr(optim, 'get_config', lambda: {})()
    except:
        conf = {}
    obj = pd.Series(conf)
    obj['name'] = name
    obj['learning_rate'] = lr
    obj.to_json(optim_path)
    print(f'Saved {name} config to {optim_path} ✓')
    args.optimizer = optim

    ####################### Callbacks  #######################
    
    epoch_loss = EpochLoss(os.path.join(out_dir, 'epoch_loss.csv'))


    ####################### Train model #######################

    seed(1)
    set_random_seed(2)
    my_model = train_model(args.data_path, args,
        callbacks=[epoch_loss], weights=args.weights)

    ####################### Save output #######################

    my_model.save(os.path.join(out_dir, 'model.h5'), include_optimizer=False)
