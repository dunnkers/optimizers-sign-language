import unittest
from combine_datasets import combine_datasets
import os

class CombineDatasets(unittest.TestCase):
    def test_combine_datasets(self):
        df = combine_datasets('./test_data', class_encoding='file')
        self.assertTrue(all(
            df.columns.to_numpy() == ['filepath', 'filename', 'class']
        ))
        self.assertTrue(df['class'].str.match(r'[A-Z]').all())
        self.assertTrue(df['filepath'].map(lambda x: os.path.isabs(x)).all())

if __name__ == '__main__':
    unittest.main()
