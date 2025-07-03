import os
from PIL import Image
import numpy as np


def convert_masks_to_bw(input_dir, output_dir):
    """
    Converts RGB mask images in a nested directory structure to black-and-white masks.
    Black for the background and white for the foreground, ignoring hidden files and folders.

    Parameters:
        input_dir (str): Path to the input directory containing directories of images.
        output_dir (str): Path to the output directory where processed images will be saved.
    """
    folder_counter = 0
    image_counter = 0
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for file in files:
            # Ignore hidden files
            if file.startswith("."):
                continue

            # Full path to the input file
            input_path = os.path.join(root, file)

            # Calculate the relative path and create corresponding output directory
            relative_path = os.path.relpath(root, input_dir)
            output_subdir = os.path.join(output_dir, relative_path)
            os.makedirs(output_subdir, exist_ok=True)
            print(output_subdir)
            folder_counter += 1

            # Full path to the output file
            output_path = os.path.join(output_subdir, file)
            image_counter += 1


            try:
                # Open the image
                img = Image.open(input_path)

                # Convert to numpy array and process
                img_array = np.array(img)
                bw_mask = np.where(img_array > 0, 255, 0).astype(np.uint8)  # Foreground white, background black

                # Save the black-and-white mask
                bw_image = Image.fromarray(bw_mask)
                bw_image.save(output_path)
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

    print(f'number of folders: {folder_counter}, number of images: {image_counter}')


def count_subdirectories_and_images(directory):
    # Initialize counters
    folder_counter = 0
    total_image_counter = 0

    # Iterate through subdirectories in the given directory
    for root, dirs, files in os.walk(directory):
        # Ignore hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        if root != directory:  # Skip the top-level directory
            folder_counter += 1
            image_counter = 0

            # Count non-hidden image files in the current subdirectory
            for file in files:
                if not file.startswith('.') and file.lower().endswith(
                        ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    image_counter += 1

            total_image_counter += image_counter
            print(f'Subdirectory: {os.path.basename(root)}, number of images: {image_counter}')

    print(f'Total subdirectories: {folder_counter}, total number of images: {total_image_counter}')
    return folder_counter, total_image_counter


# Example usage
directory_path = "/Users/leila/Desktop/medvt/dataset/Youtube2019/train/Annotations"
count_subdirectories_and_images(directory_path)

# # Example Usage
# input_directory = "/Users/leila/Desktop/medvt/dataset/Youtube2019/train/Annotations"
# output_directory = "/Users/leila/Desktop/medvt/dataset/Youtube2019/train/Annotations"
# convert_masks_to_bw(input_directory, output_directory)