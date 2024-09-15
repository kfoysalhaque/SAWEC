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