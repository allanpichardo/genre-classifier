from __future__ import absolute_import, division, print_function, unicode_literals

import datetime
import os
import time

import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
import zipfile
from datagenerator import DataSequence
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D, LSTM
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_DIR = "dataset"

cwd = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.join(cwd, DATASET_DIR, 'mfcc_fma_small')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

batch_size = 8
classes = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']


def get_train_generator():
    train_data_gen = DataSequence(train_dir, batch_size, shuffle=True)
    return train_data_gen


def get_validation_generator():
    val_data_gen = DataSequence(val_dir, batch_size)
    return val_data_gen


def get_test_generator():
    val_data_gen = DataSequence(test_dir, batch_size)
    return val_data_gen


def get_input_shape(generator: DataSequence):
    shape = generator[0][0][0].shape
    return shape


def get_model(input_shape):
    model = Sequential()

    model.add(Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Dropout(0.2))
    model.add(Conv1D(128, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False, activation='relu'))

    # model.add(Flatten())

    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(8, activation='softmax'))

    sgd = tf.keras.optimizers.SGD(lr=0.003, decay=1e-6, momentum=0.9, nesterov=True, clipvalue=1.0)

    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def main():
    checkpoint_file = os.path.join(cwd, 'checkpoints', 'genre.best.hdf5')

    ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H_%M_%S')
    if not os.path.exists(os.path.join(cwd, 'log_dir', ts)):
        os.makedirs(os.path.join(cwd, 'log_dir', ts))
    log_dir = os.path.join(cwd, 'log_dir', ts)

    model_file = os.path.join(cwd, 'models', 'genre.h5')

    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_images=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                                                     patience=10, min_lr=0.0001)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

    epochs = 1000

    train_data_gen = get_train_generator()
    val_data_gen = get_validation_generator()
    test_data_gen = get_test_generator()

    model = get_model(get_input_shape(train_data_gen))
    print(model.summary())

    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=train_data_gen.__len__(),
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=train_data_gen.__len__(),
        callbacks=[early_stop, reduce_lr, checkpoint, tensorboard]
    )

    model.load_weights(checkpoint_file)
    model.save(model_file)

    metrics = model.evaluate_generator(
        test_data_gen,
        steps=train_data_gen.__len__(),
        verbose=1
    )

    print(metrics)


if __name__=="__main__":
    main()