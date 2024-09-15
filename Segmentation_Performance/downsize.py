import os
import cv2
import argparse
from PIL import Image
import math


def get_subdirectories(directory):
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return subdirectories


def compress_images(input_folder, output_folder, ratio=16):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the image
            with Image.open(input_path) as img:
                # Compress the image with the specified compression level
                # Save the compressed image
                img.save(output_path, "JPEG", quality=int(100/ratio), optimize=False)
                # img.save(output_path, quality=100 // ratio)


def reshape_images(input_folder, output_folder, ratio):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    for image_file in image_files:
        # Read the image
        image_path = os.path.join(input_folder, image_file)
        original_image = cv2.imread(image_path)

        if original_image is not None:
            # Get the original image shape
            original_shape = original_image.shape[:2]

            # Calculate the new shape
            new_shape = (original_shape[1] // ratio, original_shape[0] // ratio)

            # Resize the image
            reshaped_image = cv2.resize(original_image, new_shape)

            # Save the reshaped image to the output folder
            output_path = os.path.join(output_folder, f"{image_file}")
            cv2.imwrite(output_path, reshaped_image)

            print(f"Reshaped {image_file} from {original_shape} to {new_shape} and saved at {output_path}")


if __name__ == '__main__':
    # Create a command-line argument parser
    parser = argparse.ArgumentParser(description=__doc__)

    # Define command-line arguments
    parser.add_argument('environment', help='name of the environment, "Classroom1" or "Anechoic1"')
    parser.add_argument('conversion_type', help='name the conversion type, either "downsize", or "compression"')
    parser.add_argument('ratio', help='doiwnsize or compression ratio', type=float)

    args = parser.parse_args()

    # Set variables based on command-line arguments
    environment = args.environment
    conversion_type = args.conversion_type
    ratio_name = args.ratio
    ratio = math.floor(1/float(ratio_name))
    # print(environment)
    print(ratio)

    base_dir = 'val_data/' + environment
    partial_frame_dir_10k = base_dir + '_10K/'
    partial_frame_dir_downsize_half = base_dir + '_downsize_' + str(ratio_name)
    partial_frame_dir_compression_one_eighth = base_dir + 'compression_' + str(ratio_name)

    # Get subdirectories
    subdirectories = get_subdirectories(partial_frame_dir_10k)

    if conversion_type == 'downsize':
        for subdir in subdirectories:
            current_partial_frame_dir_10k = os.path.join(partial_frame_dir_10k, subdir)
            current_partial_frame_dir_downsize_half = os.path.join(partial_frame_dir_downsize_half, subdir)
            if not os.path.exists(current_partial_frame_dir_downsize_half):
                os.makedirs(current_partial_frame_dir_downsize_half)
            reshape_images(current_partial_frame_dir_10k, current_partial_frame_dir_downsize_half, ratio)

    elif conversion_type == 'compression':
        for subdir in subdirectories:
            current_partial_frame_dir_10k = os.path.join(partial_frame_dir_10k, subdir)
            current_partial_frame_dir_compression_one_eighth = os.path.join(partial_frame_dir_compression_one_eighth, subdir)
            if not os.path.exists(current_partial_frame_dir_compression_one_eighth):
                os.makedirs(current_partial_frame_dir_compression_one_eighth)
            compress_images(current_partial_frame_dir_10k, current_partial_frame_dir_compression_one_eighth, ratio)
