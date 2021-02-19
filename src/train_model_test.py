import unittest
from dataset import getdataset
from combine_datasets import combine_datasets
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Input
import json
import os
from time_history import TimeHistory
from keras.metrics import top_k_categorical_accuracy

class TestTrainModel(unittest.TestCase):
    def setUp(self):
        self.data = combine_datasets('./test_data', class_encoding='file')
        self.dataset = getdataset(self.data)

    def test_modelfit(self):
        self.dataset = self.dataset.cache().prefetch(tf.data.AUTOTUNE)

        # model setup
        time_callback = TimeHistory()
        model = tf.keras.applications.MobileNetV3Small(
            weights=None, classes=26)
        model.compile(
            optimizer='adam', 
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
            metrics=['accuracy',
                'categorical_accuracy',
                top_k_categorical_accuracy])
        print(model.summary())

        # train model
        hist = model.fit(self.dataset, epochs=1,
            callbacks=[time_callback])
        hist.history['epoch_time'] = time_callback.times

        # save model
        model.save(os.path.join('models', 'test'))
        filepath = os.path.join('models', 'test', 'history.json')
        with open(filepath, 'w') as f:
            json.dump(hist.history, f)
    
    def test_inputmodelfit(self):
        """Works the same as above, but different functions."""
        ds_train = getdataset(self.data,
            seed=343,
            validation_split=0.1,
            subset='training').cache().prefetch(tf.data.AUTOTUNE)
        ds_test = getdataset(self.data,
            seed=343,
            validation_split=0.1,
            subset='validation').cache().prefetch(tf.data.AUTOTUNE)

        i = Input((224, 224, 3))
        x = tf.keras.applications.mobilenet_v3.preprocess_input(i)
        x = MobileNetV3Small(input_tensor=x, classes=26, weights=None)(x)
        model = tf.keras.Model(inputs=i, outputs=x)

        model.compile(
            optimizer='adam', 
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
            metrics=[
                'accuracy',
                'categorical_accuracy',
                top_k_categorical_accuracy])
        print(model.summary())
        hist = model.fit(ds_train, validation_data=ds_test, epochs=1)

if __name__ == '__main__':
    unittest.main()
