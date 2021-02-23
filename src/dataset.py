import pandas as pd
from tensorflow.python.keras.preprocessing.image_dataset import paths_and_labels_to_dataset
from tensorflow.python.keras.preprocessing import dataset_utils
from tensorflow.python.keras.layers.preprocessing import image_preprocessing
from keras.utils import to_categorical
import numpy as np

def getdataset(data,
               label_mode='int',
               num_channels=3,
               batch_size=32,
               input_shape=(224, 224),
               shuffle=True,
               seed=None,
               validation_split=None,
               subset=None,
               interpolation='bilinear'
               ):
    """
    Formats a dataset described in .csv to a tf.data.Dataset object. Looks
    like `keras.preprocessing.image_dataset_from_directory`, but allows custom
    image paths.
    `data` must be a Pandas DataFrame with columns:
        `filepath`: path to image (string)
        `class`:    label for sample (string)
    
    Returns: a tf.data.Dataset object
    """
    # Check args
    interpolation = image_preprocessing.get_interpolation(interpolation)
    dataset_utils.check_validation_split_arg(
        validation_split, subset, shuffle, seed)
    
    # Load data from DataFrame
    image_paths = data['filepath'].values
    num_classes = len(data['class'].unique())
    labels, class_names = pd.factorize(data['class'])
    class_names = class_names.sort_values()
    labels = to_categorical(labels, dtype='int8') # to one-hot-encoded vector

    # Shuffle all filepaths and labels
    if shuffle:
        # Shuffle globally to erase macro-structure
        if seed is None:
            seed = np.random.randint(1e6)
        rng = np.random.RandomState(seed)
        rng.shuffle(image_paths)
        rng = np.random.RandomState(seed)
        rng.shuffle(labels)

    # CV split
    image_paths, labels = dataset_utils.get_training_or_validation_split(
        image_paths, labels, validation_split, subset)

    # Construct dataset
    dataset = paths_and_labels_to_dataset(
        image_paths,
        input_shape,
        num_channels,
        labels,
        label_mode,
        num_classes,
        interpolation
    )

    # Shuffle and batch
    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
    dataset = dataset.batch(batch_size)
    dataset.class_names = class_names.values
    dataset.file_paths = image_paths

    print(f'Loaded {len(image_paths)} files from {num_classes} categories.')
    return dataset