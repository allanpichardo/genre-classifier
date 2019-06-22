from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
import zipfile
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ZIP_DIR = "datasets/spectro_fma_small.zip"
DATASET_DIR = "dataset/"


def separate_dataset(base_dir, classes):
    for cl in classes:
        img_path = os.path.join(base_dir, cl)
        images = glob.glob(img_path + '/*.csv')
        print("{}: {} Images".format(cl, len(images)))
        num_train = int(round(len(images)*0.8))
        train, val_tmp = images[:num_train], images[num_train:]
        num_test = int(round(len(val_tmp)*0.5))
        val, test = val_tmp[:num_test], val_tmp[num_test:]

        for t in train:
            if not os.path.exists(os.path.join(base_dir, 'train', cl)):
                os.makedirs(os.path.join(base_dir, 'train', cl))
            shutil.move(t, os.path.join(base_dir, 'train', cl))

        for v in val:
            if not os.path.exists(os.path.join(base_dir, 'val', cl)):
                os.makedirs(os.path.join(base_dir, 'val', cl))
            shutil.move(v, os.path.join(base_dir, 'val', cl))

        for ts in test:
            if not os.path.exists(os.path.join(base_dir, 'test', cl)):
                os.makedirs(os.path.join(base_dir, 'test', cl))
            shutil.move(ts, os.path.join(base_dir, 'test', cl))


def main():
    cwd = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(cwd, DATASET_DIR, 'mfcc_fma_small')
    classes = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']
    separate_dataset(base_dir, classes)


if __name__=="__main__":
    main()