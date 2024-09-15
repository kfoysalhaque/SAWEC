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

import cv2
import numpy as np


def resize_image(images, target_size=(1024, 1024)):
    """
    Resize a list of images to the specified target size.

    Parameters:
    - images: NumPy array of shape (num_images, height, width, channels)
    - target_size: Tuple specifying the target size (height, width)

    Returns:
    - NumPy array of resized images with shape (num_images, target_size[0], target_size[1], channels)
    """
    resized_images = []

    for img in images:
        # Remove the single channel dimension
        img = img.squeeze()

        # Resize the image to the specified target size
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

        # Add the single channel dimension back
        resized_img = resized_img[:, :, np.newaxis]

        # Append the resized image to the list
        resized_images.append(resized_img)

    # Convert the list of resized images back to a NumPy array
    resized_images = np.array(resized_images)

    return resized_images