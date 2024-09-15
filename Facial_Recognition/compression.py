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
from PIL import Image
import tensorflow as tf
import os
import time
import argparse


# Set GPU visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.list_physical_devices('GPU')
print("Available GPUs:", physical_devices)

# Check if TensorFlow is using GPU
if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
    print("TensorFlow is using GPU acceleration.")
else:
    print("TensorFlow is not using GPU acceleration.")


def compress_image(input_path, output_path, quality=6):
    """
    Compresses an image and saves the compressed image.

    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the compressed image.
        quality (int): Compression quality (0-100).

    Returns:
        bool: True if compression succeeds, False otherwise.
        float: Compression time in seconds.
    """
    try:
        # Open the image
        image = Image.open(input_path)

        # Correct the orientation of the image (rotate it 270 degrees)
        image = image.rotate(270, expand=True)

        # Measure compression time
        start_time = time.time()

        # Save the compressed image
        image.save(output_path, "JPEG", quality=quality, optimize=False)

        end_time = time.time()
        compression_time = end_time - start_time

        return True, compression_time
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False, 0

def create_output_folders(source_folder, destination_folder):
    """
    Creates output folders based on the structure of the source folder.

    Args:
        source_folder (str): Path to the source folder.
        destination_folder (str): Path to the destination folder.
    """
    for root, _, _ in os.walk(source_folder):
        relative_path = os.path.relpath(root, source_folder)
        output_dir = os.path.join(destination_folder, relative_path)
        os.makedirs(output_dir, exist_ok=True)


def compress_images_in_folder(source_folder, destination_folder, quality=6):
    """
    Compresses images in a folder and saves the compressed images to a destination folder.

    Args:
        source_folder (str): Path to the source folder containing images.
        destination_folder (str): Path to the destination folder to save compressed images.
        quality (int): Compression quality (0-100).
    """
    create_output_folders(source_folder, destination_folder)
    total_compression_time = 0
    failed = 0
    succeed = 0

    for root, _, files in os.walk(source_folder):
        relative_path = os.path.relpath(root, source_folder)

        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                input_image_path = os.path.join(root, filename)
                output_image_path = os.path.join(destination_folder, relative_path, filename)

                success, compression_time = compress_image(input_image_path, output_image_path, quality)
                if success:
                    total_compression_time += compression_time
                    succeed += 1
                else:
                    failed += 1

    print(f"Total compression time for {succeed} images: {total_compression_time:.2f} seconds")
    print(f'Total unsuccessful attempts: {failed}')


if __name__ == "__main__":
    # Create a command-line argument parser
    parser = argparse.ArgumentParser(description=__doc__)

    # Define command-line arguments
    parser.add_argument('compression_quality', help='compression quality of the image (0-100), higher the value, better the image', type=int)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Set variables based on command-line arguments
    compression_quality = int(args.compression_quality)
    source_folder = "Test_Data"  # Replace with your source folder path
    destination_folder = 'compression_quality_' + str(compression_quality)  # Replace with your destination folder path
    compress_images_in_folder(source_folder, destination_folder, compression_quality)