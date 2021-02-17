import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image_dataset import paths_and_labels_to_dataset

def getdataset(data):
    """
    `data` must be a Pandas DataFrame with columns:
        `filepath`: path to image (string)
        `class`:    label for sample (string)
    
    Returns: a tf.data.Dataset object
    """
    image_paths = data['filepath'].values
    classes, class_names = pd.factorize(data['class'].sort_values())
    num_classes = len(data['class'].unique())

    labels = tf.keras.utils.to_categorical(
        classes,
        num_classes=num_classes,
        dtype='uint8'
    )

    dataset = paths_and_labels_to_dataset(
        image_paths,
        (200, 200),
        3,
        labels,
        'categorical',
        num_classes,
        'bilinear'
    )

    batch_size = 32
    seed = 343
    dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
    dataset = dataset.batch(batch_size)
    dataset.class_names = class_names.values
    dataset.file_paths = image_paths

    print(f'Loaded {len(image_paths)} files from {num_classes} categories.')
    return dataset