from PIL import Image
import numpy as np


def extract_image_section(image_path, crop_save, pixels_per_degree, theta, cluster_range, alpha_width=1.5, alpha_height=2.5):

    # Load the image
    original_image = Image.open(image_path)
    center_width = int(pixels_per_degree * theta)
    center_height = 2400
    image_half = int((pixels_per_degree * cluster_range))

    # Define the pixel ranges for the section
    start_width = int(center_width - image_half * alpha_width)
    if start_width < 0:
        start_width = 0
    if start_width > 9600:
        start_width = 9600
    end_width = int(center_width + image_half * alpha_width)

    if end_width > 9600:
        end_width = 9600
    if end_width < 0:
        end_width = 0

    start_height = int(center_height - image_half * alpha_height)
    end_height = int(center_height + image_half * alpha_height)

    # Convert the image to a NumPy array
    image_array = np.array(original_image)

    # Take the specified section
    section = image_array[start_height:end_height, start_width:end_width, :]

    # Convert the NumPy array back to a PIL image
    section_image = Image.fromarray(section)

    # Save or display the section image
    # section_image.show()

    # Get the size (width and height) of the cropped image
    # cropped_width, cropped_height = section_image.size
    # print(cropped_width, cropped_height)
    # Or save it to a file
    section_image.save(crop_save)
