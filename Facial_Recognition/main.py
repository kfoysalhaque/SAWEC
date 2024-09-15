"""
Copyright (C) 2024 Khandaker Foysal Haque
contact: haque.k@northeastern.edu
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import division
import numpy as np
import tensorflow as tf
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
import os
import argparse
from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.vgg8 import vgg8


def compile_model(model):
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy'])


def train_model(model, train_data, train_label, valid_data, valid_label, batch_size=16, epochs=10, model_checkpoint_path='Z.h5'):
    model_checkpoint = ModelCheckpoint(model_checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, mode='max', verbose=1)

    if batch_size is None or not batch_size:  # Use "or not batch_size" to check for None or 0
        batch_size = 16  # Set a default batch size if it is None or 0

    model.fit(train_data, train_label,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(valid_data, valid_label),
              shuffle=True,
              callbacks=[model_checkpoint, early_stopping]
              )



if __name__ == '__main__':


    # Set variables based on command-line arguments
    test_name = 'downsize'
    print(test_name)
    compression_quality_or_image_resolution = 1024
    print(compression_quality_or_image_resolution)

    # Check if compression_quality is valid
    valid_compression_values = {0, 6, 12, 25, 50, 75, 100}
    valid_image_resolutions = {1024, 512, 256, 128, 64, 32}


    if test_name == 'compression':
        compression_quality = compression_quality_or_image_resolution
        width, height = 1024, 1024
        if compression_quality not in valid_compression_values:
            print("Error: Compression quality must be one of 0, 25, 50, 75, or 100.")
            exit(1)
    elif test_name == 'downsize':
        image_resolution = compression_quality_or_image_resolution
        if image_resolution not in valid_image_resolutions:
            print("Error: Image resolution must be one of 1024, 512, 256, 128, 64, 32")
            exit(1)

    if test_name == 'compression':
        test_directory = 'Compressed_Data' + str(compression_quality) + '/'
    elif test_name == 'downsize':
        test_directory = str(image_resolution) + 'x' + str(image_resolution) + '/'

    best_model = 'facial_recognition.h5'

    train_data = test_directory + 'X_train.npy'
    train_label = test_directory + 'y_train.npy'

    valid_data = test_directory + 'X_valid.npy'
    valid_label = test_directory + 'y_valid.npy'

    test_data = test_directory + 'X_test.npy'
    test_label = test_directory + 'y_test.npy'

    train_data = np.array(np.load(train_data))
    train_label = np.array(np.load(train_label))

    valid_data = np.array(np.load(valid_data))
    valid_label = np.array(np.load(valid_label))

    test_data = np.load(test_data)
    test_label = np.load(test_label)


    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)

    # Check if TensorFlow is using GPU
    if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
        print("TensorFlow is using GPU acceleration.")
    else:
        print("TensorFlow is not using GPU acceleration.")

    num_features = 32
    num_labels = 6
    batch_size = 2 #16
    epochs = 200
    width, height = image_resolution, image_resolution
    input_shape = (width, height, 1)

    # Build the model
    model = vgg8(input_shape, num_labels)

    # Compile the model
    compile_model(model)

    # Train the model
    train_model(model, train_data, train_label, valid_data, valid_label, epochs=epochs, batch_size=batch_size, model_checkpoint_path =best_model)
