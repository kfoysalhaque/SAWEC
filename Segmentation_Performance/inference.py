import os
import time
import argparse
from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO('yolov8x-seg.pt')

# Display model information (optional)
model.info()


if __name__ == '__main__':
    # Create a command-line argument parser
    parser = argparse.ArgumentParser(description=__doc__)

    # Define command-line arguments
    parser.add_argument('frame_directory', help='name of the frame_directory, "Frames"'
                                                'or "Partial_Frames_10K" '
                                                'or "Partial_Frames_Compression_0.06" '
                                                'or "Partial_Frames_Downsize_0.5"')

    args = parser.parse_args()

    # Set variables based on command-line arguments
    frame_dir = args.frame_directory
    base_dir = os.path.join('../Stitched_Video/')
    envs = ['Anechoic1', 'Classroom1']
    # Run inference on images in subdirectories
    start_time = time.time()

    for env in envs:
        image_dir = os.path.join(base_dir, env, frame_dir)
        # Loop through subdirectories
        for subdir, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):  # Assuming images have these extensions
                    image_path = os.path.join(subdir, file)
                    # Run inference on each image
                    results = model(image_path, save=True, imgsz=640)

        inference_time = time.time() - start_time
        print(f"Time taken for inference: {inference_time:.2f} seconds")
