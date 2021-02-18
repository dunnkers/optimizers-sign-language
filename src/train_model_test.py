import unittest
from dataset import getdataset
from combine_datasets import combine_datasets
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Input
import json

class TestTrainModel(unittest.TestCase):
    def setUp(self):
        self.data = combine_datasets('./test_data', class_encoding='file')
        self.dataset = getdataset(self.data)

    def test_modelfit(self):
        self.dataset = self.dataset.cache().prefetch(tf.data.AUTOTUNE)

        # model setup
        model = tf.keras.applications.MobileNetV3Small(
            weights=None, classes=26)
        model.compile(
            optimizer='adam', 
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
            metrics=['accuracy'])
        print(model.summary())

        # train model
        hist = model.fit(self.dataset, epochs=1)

        # save model
        model.save('trainedmodel')
        with open('trainedmodel'+"_history.txt", 'w') as f:
            json.dump(hist.history, f)

        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
