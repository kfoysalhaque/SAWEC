from PIL import Image
import os
import numpy as np
import math
import numpy as np
import argparse
from sklearn.cluster import DBSCAN
from matplotlib.patches import Ellipse
from modules.translate_aoa import translate_aoa_to_360_frame_angle
from modules.extract_image_section import extract_image_section
from modules.dbscan import dbscan


def get_subdirectories(directory):
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return subdirectories


if __name__ == '__main__':
    # Create a command-line argument parser
    parser = argparse.ArgumentParser(description=__doc__)

    # Define command-line arguments
    parser.add_argument('environment', help='name of the environment, "Classroom1" or "Anechoic1"')
    args = parser.parse_args()

    # Set variables based on command-line arguments
    environment = args.environment
    print(environment)

    # Given values
    AC = 3
    theta_tx = 335
    Ax = 0.3
    Bx = 0
    pixels_per_degree = 26.67

    base_dir = '../Stitched_Video/' + environment
    image_dir = os.path.join(base_dir, 'Frames')
    loc_dir = os.path.join(base_dir, 'Localizations')
    partial_frame_dir = os.path.join(base_dir, 'Partial_Frames_10K')

    # Check if the directory exists
    if not os.path.exists(partial_frame_dir):
        # If not, create the directory
        os.makedirs(partial_frame_dir)

    # Get subdirectories
    subdirectories = get_subdirectories(image_dir)

    for subdir in subdirectories:
        current_image_dir = os.path.join(image_dir, subdir)
        current_loc_dir = os.path.join(loc_dir, subdir)
        current_partial_frame_dir = os.path.join(partial_frame_dir, subdir)
        if not os.path.exists(current_partial_frame_dir):
            os.makedirs(current_partial_frame_dir)

        localization_files = [f for f in os.listdir(current_loc_dir) if f.lower().endswith('.npy')]

        for loc_file in localization_files:
            loc_file = os.path.join(current_loc_dir, loc_file)
            print(subdir, loc_file)
            string_num = loc_file[-8:-4]
            image_file = current_image_dir + '/frame_' + string_num + '.png'

            # DBSCAN
            clusters, cluster_labels = dbscan(loc_file)
            # Separate out the clusters
            unique_labels = np.unique(cluster_labels)
            ellipse_data = []  # To store AoA and corresponding heights

            print(string_num)

            for i, label in enumerate(unique_labels):
                if label == -1:
                    # Skip noise points
                    continue
                cluster_points = clusters[cluster_labels == label]
                # clusters.append(cluster_points)

                # Fit ellipse to the cluster
                ellipse = Ellipse(np.mean(cluster_points, axis=0),
                                  width=np.std(cluster_points[:, 0]) * 2,
                                  height=np.std(cluster_points[:, 1]) * 2,
                                  edgecolor='black', facecolor='none')

                partial_frame_file = current_partial_frame_dir + '/' + subdir + '_' + string_num + '_' + str(i) + '.png'

                AoA = ellipse.center[1]
                print(AoA)
                print(ellipse.height)
                cluster_range = ellipse.height * 1.5  # compensate for DBSCAN reduction of range

                # AoA Correction and AoA Projection to 360 Frame Scale
                corrected_AoA, distance, theta = translate_aoa_to_360_frame_angle(Ax, Bx, AC, theta_tx, AoA)
                print(corrected_AoA)
                print(theta)

                # Extract image section based on AoA Projection
                extract_image_section(image_file, partial_frame_file, pixels_per_degree, theta, cluster_range)