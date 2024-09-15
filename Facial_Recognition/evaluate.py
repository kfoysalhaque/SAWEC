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

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
import os
import time
from tensorflow.keras.models import load_model
import argparse
import cv2
from utils.evaluate_performance import evaluate_performance
from utils.resize_image import resize_image


if __name__ == '__main__':
    # Create a command-line argument parser
    parser = argparse.ArgumentParser(description=__doc__)

    # Define command-line arguments
    parser.add_argument('test_name', help='test name- "compression" / "downsize"')
    parser.add_argument('compression_quality_or_image_resolution',
                        help='compression quality (0, 25, 50, 75, 100), higher the value, better the image quality'
                             '  For image downsize test, input image resolution (1024/512/256/128/64/32)')
    args = parser.parse_args()

    # Set variables based on command-line arguments
    test_name = args.test_name
    print(test_name)
    compression_quality_or_image_resolution = int(args.compression_quality_or_image_resolution)
    print(compression_quality_or_image_resolution)

    # Check if compression_quality is valid
    valid_compression_values = {0, 1, 3, 6, 12, 25, 50, 75, 100}
    valid_image_resolutions = {1024, 512, 256, 128, 64, 32}
    width, height = 1024, 1024

    if test_name == 'compression':
        compression_quality = compression_quality_or_image_resolution
        if compression_quality not in valid_compression_values:
            print("Error: Compression quality must be one of 0, 25, 50, 75, or 100.")
            exit(1)
    elif test_name == 'downsize':
        image_resolution = compression_quality_or_image_resolution
        if image_resolution not in valid_image_resolutions:
            print("Error: Image resolution must be one of 1024, 512, 256, 128, 64, 32")
            exit(1)

    if test_name == 'compression':
        test_directory = 'compression_quality_' + str(compression_quality) + '/'
    elif test_name == 'downsize':
        test_directory = str(image_resolution) + 'x' + str(image_resolution) + '/'

    best_model = 'facial_recognition.h5'

    test_data = test_directory + 'X_test.npy'
    test_label = test_directory + 'y_test.npy'

    test_data = np.load(test_data)
    test_label = np.load(test_label)

    if test_name == 'downsize' and image_resolution != 1024:
        test_data = resize_image(test_data, target_size=(width, height))

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)

    # Check if TensorFlow is using GPU
    if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
        print("TensorFlow is using GPU acceleration.")
    else:
        print("TensorFlow is not using GPU acceleration.")

    # Load the model
    model = load_model(best_model)
    evaluate_performance(model, test_data, test_label)
