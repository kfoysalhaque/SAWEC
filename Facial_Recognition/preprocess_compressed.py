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
import argparse
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from sklearn.model_selection import train_test_split
import shutil
from utils.augmentation import augmentation
from utils.resize_image_in_dir import resize_image_in_dir

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)

# Check if TensorFlow is using GPU
if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
    print("TensorFlow is using GPU acceleration.")
else:
    print("TensorFlow is not using GPU acceleration.")

width, height = 1024, 1024

def normalize_image(image):
    """
    Normalize an image by scaling its pixel values to the range [0, 1].

    Args:
        image: Input image as a NumPy array.

    Returns:
        Normalized image as a NumPy array.
    """
    min_value = np.min(image)
    max_value = np.max(image)
    normalized_image = (image - min_value) / (max_value - min_value)

    return normalized_image


def delete_subdirectories(directory_path):
    # Get the list of all items (files and directories) in the given directory
    items = os.listdir(directory_path)

    # Iterate over the items
    for item in items:
        item_path = os.path.join(directory_path, item)

        # Check if it's a directory
        if os.path.isdir(item_path):
            # Remove the directory (recursively)
            try:
                shutil.rmtree(item_path)
                print(f"Deleted subdirectory: {item_path}")
            except Exception as e:
                print(f"Failed to delete subdirectory {item_path}: {e}")


if __name__ == "__main__":
    # Create a command-line argument parser
    parser = argparse.ArgumentParser(description=__doc__)

    # Define command-line arguments
    parser.add_argument('compression_quality',
                        help='compression quality of the image (0-100), higher the value, better the image', type=int)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Set variables based on command-line arguments
    compression_quality = int(args.compression_quality)
    source_folder = "Test_Data"  # Replace with your source folder path
    destination_folder = 'compression_quality_' + str(compression_quality)  # Replace with your destination folder path
    train_data = destination_folder + "/X_train"
    train_label = destination_folder + "/y_train"
    test_data = destination_folder + "/X_test"
    test_label = destination_folder + "/y_test"
    valid_data = destination_folder + "/X_valid"
    valid_label = destination_folder + "/y_valid"

    pixels, labels = resize_image_in_dir(destination_folder)

    # Converting list into numpy array
    X = np.asarray(pixels)
    # R_x = np.asarray(resized_pixels)

    # Normalize
    Xn = normalize_image(X)
    # R_xn = normalize_image(R_x)

    # Expand Dimension
    Xr = np.expand_dims(Xn, -1)
    # R_xr = np.expand_dims(R_xn, -1)

    # Reshaping Label and onehot encoding
    string_array = np.array(labels).reshape(-1, 1)
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(string_array)
    Y = Y.toarray()

    # counting classes
    column_counts = np.count_nonzero(Y == 1, axis=0)
    print("Column count", column_counts)

    Xr, Y = augmentation(Xr, Y)

    # splitting into training, validation and testing data
    X_train, X_test, y_train, y_test = train_test_split(Xr, Y, test_size=0.15, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=41)

    # counting test classes
    column_counts_train = np.count_nonzero(y_train == 1, axis=0)
    print("Column count train", column_counts_train)
    # column_counts_train_resize = np.count_nonzero(Ry_train == 1, axis=0)

    column_counts_test = np.count_nonzero(y_test == 1, axis=0)
    print("Column count test", column_counts_test)
    # column_counts_test_resize = np.count_nonzero(Ry_test == 1, axis=0)

    # X_train, y_train = augmentation(X_train, y_train)

    # Print the shapes of the augmented data
    print("Augmented Images Shape:", X_train.shape)
    print("Augmented Labels Shape:", y_train.shape)

    # counting augmented test classes
    column_count_aug = np.count_nonzero(y_train == 1, axis=0)

    # storing them using numpy
    np.save(train_data, X_train)
    np.save(train_label, y_train)
    np.save(valid_data, X_valid)
    np.save(valid_label, y_valid)
    np.save(test_data, X_test)
    np.save(test_label, y_test)
    delete_subdirectories(destination_folder)

