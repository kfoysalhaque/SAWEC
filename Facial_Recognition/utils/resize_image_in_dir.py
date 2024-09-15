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


def resize_image_in_dir(base_directory, new_size=(1024, 1024)):
    pixels = []
    labels = []

    # Iterate over subdirectories within the base_directory
    for subdir in os.listdir(base_directory):
        subbase_directory = os.path.join(base_directory, subdir)

        # Check if the item is a base_directory
        if os.path.isdir(subbase_directory):
            # Assign a label based on the subbase_directory name
            label = subdir

            # Iterate over all files in the subbase_directory
            for filename in os.listdir(subbase_directory):
                # Create the file path
                filepath = os.path.join(subbase_directory, filename)

                # Read the image using OpenCV
                image = cv2.imread(filepath)

                # Convert the image to grayscale
                grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Creating a list of original image
                # pixel_value = cv2.resize(grayscale_image, (base_width, base_height))
                pixel_value = cv2.resize(grayscale_image, (new_size[0], new_size[1]))
                pixels.append(pixel_value.astype('float32'))

                # Resizing image
                # resized_image = imutils.resize(grayscale_image, width=new_size[0], height=new_size[1])

                # Creating a list of resized value
                # resized_pixel_values.append(resized_image.astype('float32'))

                # Checking condition
                if label == "Angry":
                    lb = 0
                elif label == "Disgust":
                    lb = 1
                elif label == "Happy":
                    lb = 2
                elif label == "Sad":
                    lb = 3
                elif label == "Surprise":
                    lb = 4
                else:
                    lb = 5
                # Assign label to the corresponding image
                labels.append(lb)

    return pixels, labels

