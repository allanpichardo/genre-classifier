import math
import pandas
import random
import os
import numpy as np
import tensorflow

from pandas import DataFrame
from tensorflow.keras.utils import Sequence


class DataSequence(Sequence):
    """
    Keras Sequence object to train a model on a list of csv files
    """
    def __init__(self, rootdir, batch_size, shuffle=False, class_format='categorical', classes=['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']):
        """
        df = dataframe with two columns: the labels and a list of filenames
        """

        df = DataFrame(columns=['file_names', 'label'])
        for root, subdirs, files in os.walk(rootdir):
            for subdir in subdirs:
                for r, s, f in os.walk(os.path.join(root, subdir)):
                    paths = [os.path.join(r, name) for name in f]
                    temp = DataFrame(data=paths, columns=['file_names'])
                    temp['label'] = classes.index(subdir)
                    df = df.append(temp, ignore_index=True)

        self.df = df
        self.classes = classes
        self.bsz = batch_size
        self.shuffle = shuffle
        self.n = len(df.index)
        self.indexes = random.sample(range(self.n), k=self.n)

        # Take labels and a list of image locations in memory
        self.labels = tensorflow.keras.utils.to_categorical(self.df['label'].values, num_classes=len(self.classes)) if class_format=='categorical' else self.df['label'].values

        self.file_list = self.df['file_names']

    def __len__(self):
        return int(math.floor(self.n / float(self.bsz)))

    def on_epoch_end(self):
        self.indexes = range(self.n)
        if self.shuffle:
            # Shuffles indexes after each epoch if in training mode
            self.indexes = random.sample(self.indexes, k=len(self.indexes))

    def get_batch_labels(self, idx, arr):
        # Fetch a batch of labels
        return arr[idx * self.bsz: (idx + 1) * self.bsz]

    def get_batch_features(self, arr):
        # Fetch a batch of inputs
        feats = np.array([self.read_csv_data(f) for f in arr])
        return feats

    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.bsz:(idx+1)*self.bsz]

        files_temp = np.array([self.file_list[k] for k in indexes])
        y = np.array([self.labels[k] for k in indexes])

        batch_x = self.get_batch_features(files_temp)

        return batch_x, y

    def read_csv_data(self, filename):
        df = pandas.read_csv(filename, index_col=0).fillna(0.00000000000000001)
        df = self.normalize(df)
        return df.values

    def normalize(self, df: DataFrame):
        return (df - df.min()) / (df.max() - df.mean())




if __name__=='__main__':
    DATASET_DIR = "dataset/"
    cwd = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(cwd, DATASET_DIR, 'mfcc_fma_small', 'train')

    gen = DataSequence(base_dir, 64, True)

    batch = gen[0][0][0].shape
    print(batch)
