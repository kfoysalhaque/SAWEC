import os
import glob

folder_path = "/media/foysal/SAWEC/SAWEC/Segmentation_Performance/val_data/Partial_Frames_Anechoic1_10K/images/"
output_file = "/media/foysal/SAWEC/SAWEC/Segmentation_Performance/val_data/Partial_Frames_Anechoic1_10K/val2017.txt"

# Get a list of all image files in the folder
image_files = glob.glob(os.path.join(folder_path, "*.jpg"))

# Write the list of images with their directory to a text file
with open(output_file, 'w') as file:
    for image_file in image_files:
        # Get the relative path by removing the common prefix
        relative_path = os.path.relpath(image_file, folder_path)
        # Add the "./images/" prefix to the relative path
        file.write(f"./images/{relative_path}\n")

print(f"List of images saved to: {output_file}")
