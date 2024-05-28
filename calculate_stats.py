import os
import argparse
from PIL import Image
import numpy as np

def calculate_stats(folder_paths):
    """
    Calculates mean and standard deviation along color channels for images in multiple folders,
    ensuring values are between 0 and 1.

    Args:
        folder_paths: List of paths to folders containing images.

    Returns:
        A tuple containing two 3D vectors:
            - mean: Average pixel value for each color channel across all images (normalized to 0-1).
            - std: Standard deviation of pixel values for each color channel across all images (normalized to 0-1).
    """
    scaler = 255.0
    channel_means = np.zeros((3,))
    channel_stds = np.zeros((3,))
    num_images = 0

    # Iterate through each folder
    for folder_path in folder_paths:
        print(f"🚀 Calculating stats for {folder_path} ..")
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
                continue  # Skip non-image files

            image = Image.open(os.path.join(folder_path, filename))
            image_array = np.array(image) / scaler  # Normalize image data to 0-1 range

            # Update channel means and stds
            channel_means += np.mean(image_array, axis=(0, 1))
            channel_stds += np.std(image_array, axis=(0, 1))
            num_images += 1
        print(f"🔥 Done wtih {folder_path} ")

    # Calculate overall mean and std (divide by number of images)
    if num_images > 0:
        channel_means /= num_images
        channel_stds /= num_images

    return channel_means, channel_stds

def main():
    parser = argparse.ArgumentParser(description="Calculate image statistics (mean and std) for folders.")
    parser.add_argument('folders', metavar='F', type=str, nargs='+', help='Paths to folders containing images')

    args = parser.parse_args()
    folder_paths = args.folders

    mean, std = calculate_stats(folder_paths)
    
    print(f"Mean: {mean}")
    print(f"Std: {std}")

if __name__ == "__main__":
    main()

# Usage
# python calculate_image_stats.py /path/to/folder1 /path/to/folder2 /path/to/more/folders
