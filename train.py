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

DATASET_DIR = "dataset/"

cwd = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.join(cwd, DATASET_DIR, 'mfcc_fma_small')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

batch_size = 64
classes = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']


def get_train_generator():
    image_gen_train = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=.15,
        horizontal_flip=True
    )

    train_data_gen = image_gen_train.flow_from_directory(
        batch_size=batch_size,
        directory=train_dir,
        shuffle=True,
        class_mode='categorical'
    )

    return train_data_gen


def get_validation_generator():
    image_gen_val = ImageDataGenerator(
        rescale=1./255
    )

    val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                     directory=val_dir,
                                                     class_mode='categorical')

    return val_data_gen


def get_test_generator():
    image_gen_val = ImageDataGenerator(
        rescale=1./255
    )

    val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                     directory=test_dir,
                                                     class_mode='categorical')

    return val_data_gen

def get_input_shape(generator):
    return generator[0][0][0].shape


def get_model(input_shape):
    model = Sequential()

    model.add(Conv2D(64, 3, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense(8, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def main():
    checkpoint_file = os.path.join(cwd, 'checkpoints/genre.best.hdf5')
    log_dir = os.path.join(cwd, 'log_dir')
    model_file = os.path.join(cwd, 'models/genre.h5')

    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                                                     patience=10, min_lr=0.0001)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

    epochs = 2000

    train_data_gen = get_train_generator()
    val_data_gen = get_validation_generator()
    test_data_gen = get_test_generator()

    model = get_model(get_input_shape(train_data_gen))

    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=int(np.ceil(train_data_gen.n / float(batch_size))),
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=int(np.ceil(val_data_gen.n / float(batch_size))),
        callbacks=[early_stop, reduce_lr, checkpoint]
    )

    model.load_weights(checkpoint_file)
    model.save(model_file)

    metrics = model.evaluate_generator(
        test_data_gen,
        steps=int(np.ceil(test_data_gen.n / float(batch_size))),
        verbose=1
    )

    print(metrics)


if __name__=="__main__":
    main()