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
import cv2
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import argparse
from sklearn.model_selection import train_test_split
from utils.resize_image_in_dir import resize_image_in_dir


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


def augmentation(X, Y):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,  # Rotate the image randomly by up to 20 degrees
        width_shift_range=0.1,  # Shift the image horizontally by up to 10% of its width
        height_shift_range=0.1,  # Shift the image vertically by up to 10% of its height
        shear_range=0.2,  # Apply shear transformation randomly
        zoom_range=0.2,  # Apply random zooming in or out
        horizontal_flip=True,  # Flip the image horizontally
        fill_mode='nearest'  # Fill any newly created pixels during augmentation with the nearest pixel value
    )

    # Create a generator using the flow method
    generator = datagen.flow(X, Y, batch_size=64, shuffle=True)

    # Define the number of augmented images to generate
    num_augmented_images = 8000

    augmented_images = []
    augmented_labels = []

    # Generate augmented images and labels
    while len(augmented_images) < num_augmented_images:
        # Generate a batch of augmented images and labels
        batch_images, batch_labels = generator.next()

        # Check if adding the batch will exceed the desired number of augmented images
        if len(augmented_images) + len(batch_images) > num_augmented_images:
            # Calculate the number of images needed to reach the desired count
            remaining_images = num_augmented_images - len(augmented_images)

            # Add only the required number of images from the batch
            augmented_images.extend(batch_images[:remaining_images])
            augmented_labels.extend(batch_labels[:remaining_images])

            # Exit the loop since the desired count is reached
            break

        # Add the entire batch if it doesn't exceed the desired count
        augmented_images.extend(batch_images)
        augmented_labels.extend(batch_labels)

    # Convert the augmented images and labels to NumPy arrays
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)

    return augmented_images, augmented_labels


# Check if the script is being run as the main program
if __name__ == '__main__':
    # Create a command-line argument parser
    parser = argparse.ArgumentParser(description=__doc__)

    # Define command-line arguments
    parser.add_argument('image_resolution', help='resolution of images', type=int)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Set variables based on command-line arguments
    image_resolution = int(args.image_resolution)

    base_width, base_height = 1088, 1088

    new_size = (image_resolution, image_resolution)

    base_directory = "Test_Data"
    dir_name = str(image_resolution) + 'x' + str(image_resolution) + '/'
    augmented_images = dir_name + "Augmented"
    train_data = dir_name + "X_train"
    train_label = dir_name + "y_train"
    test_data = dir_name + "X_test"
    test_label = dir_name + "y_test"
    valid_data = dir_name + "X_valid"
    valid_label = dir_name + "y_valid"

    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)

    # Check if TensorFlow is using GPU
    if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
        print("TensorFlow is using GPU acceleration.")
    else:
        print("TensorFlow is not using GPU acceleration.")

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    pixels, labels = resize_image_in_dir(base_directory, new_size=(image_resolution, image_resolution))

    # Converting list into numpy array
    X = np.asarray(pixels)

    # Normalize
    Xn = normalize_image(X)

    # Expand Dimension
    Xr = np.expand_dims(Xn, -1)

    # Reshaping Label and one-hot encoding
    string_array = np.array(labels).reshape(-1, 1)
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(string_array)
    Y = Y.toarray()

    # counting classes
    column_counts = np.count_nonzero(Y == 1, axis=0)

    Xr, Y = augmentation(Xr, Y)

    # splitting into training, validation and testing data
    X_train, X_test, y_train, y_test = train_test_split(Xr, Y, test_size=0.15, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=41)

    # counting test classes
    column_counts_train = np.count_nonzero(y_train == 1, axis=0)

    column_counts_test = np.count_nonzero(y_test == 1, axis=0)

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
