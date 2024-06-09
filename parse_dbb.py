from pycocotools.coco import COCO
import numpy as np
import cv2
import json
import os
import matplotlib.image as mpimg
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_motion_mask(kittimots_dataset_path, mask_data, file_name):
    # Extract directory path and file name
    directory, filename = os.path.split(file_name)

    # Ensure the directory exists, create it if it doesn't
    full_directory = os.path.join(kittimots_dataset_path, directory)
    os.makedirs(full_directory, exist_ok=True)

    # Save the mask image with the same name in the corresponding directory
    mask_file_path = os.path.join(full_directory, filename)

    # Convert mask data to numpy array
    mask_array = np.array(mask_data, dtype=np.uint8)

    # Save the mask image as PNG
    cv2.imwrite(mask_file_path, mask_array)

    return mask_file_path


def image_path(kittimots_training_path, file_name):
    # print('path', os.path.join(kittimots_training_path, file_name))
    return os.path.join(kittimots_training_path, file_name)


def show_images_with_mask(img, mask):
    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Show the original image
    axes[0].imshow(img)
    axes[0].axis('off')  # Optional: hide axis
    axes[0].set_title('Original Image')

    # Show the image mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].axis('off')  # Optional: hide axis
    axes[1].set_title('Mask')

    # Save the combined image as a PNG file
    # plt.savefig('combined_image.png', bbox_inches='tight', pad_inches=0)

    plt.show()
    plt.close()


def save_mask_images(kittimots_dataset_path, mask_data, file_name):
    # Extract directory path and file name
    directory, filename = os.path.split(file_name)

    # Ensure the directory exists, create it if it doesn't
    full_directory = os.path.join(kittimots_dataset_path, directory)
    os.makedirs(full_directory, exist_ok=True)

    # Save the mask image with the same name in the corresponding directory
    mask_file_path = os.path.join(full_directory, filename)

    # Convert mask data to numpy array
    mask_array = np.array(mask_data, dtype=np.uint8)

    # Scale the mask to range 0-255
    mask_scaled = mask_array * 255

    # Save the mask image as PNG
    cv2.imwrite(mask_file_path, mask_scaled)
    print("image is saved in:", mask_file_path)

    return mask_file_path


def show_masks():
    kittimots_motion_mask_path = '/Users/leila/Desktop/medvt/dataset/KITTIMOTS/annotations/motion/0000'
    # List all files in the directory
    files = os.listdir(kittimots_motion_mask_path)

    # Loop through each file in the directory
    for filename in files:
        # Check if the file is an image (you may want to refine this check based on your file types)
        if filename.endswith('.png') or filename.endswith('.jpg'):
            # Construct the full path to the image
            image_path = os.path.join(kittimots_motion_mask_path, filename)
            img_show = mpimg.imread(image_path)
            plt.imshow(img_show, cmap='gray')
            plt.title('img_id')
            plt.axis('off')
            plt.show()
            plt.close()


def create_black_image(path):
    width = 1280
    height = 720
    # Create a black image (all zeros)
    black_image = Image.new('RGB', (width, height), color=(0, 0, 0))

    # Save the image
    black_image.save(path)
    print(f"Black image saved to {path}")


def check_image_existence(bdd_rgb_training_path, bdd_gt_path):
    # Iterate through folders in the base path
    for folder_name in sorted(os.listdir(bdd_rgb_training_path)):
        if folder_name == 'b2e54795-db1f3bad':
            folder_path = os.path.join(bdd_rgb_training_path, folder_name)
        else:
            continue
        if os.path.isdir(folder_path):
            gt_folder_path = os.path.join(bdd_gt_path, folder_name)

            if not os.path.exists(gt_folder_path):
                os.makedirs(gt_folder_path)

            # Get a set of base names of images in the ground truth folder
            gt_image_basenames = {os.path.splitext(gt_image_name)[0] for gt_image_name in os.listdir(gt_folder_path)}

            # Iterate through images in the current folder
            for image_name in sorted(os.listdir(folder_path)):
                image_base_name = os.path.splitext(image_name)[0]
                corresponding_gt_paths = [os.path.join(gt_folder_path, image_base_name + ext) for ext in
                                          ['.png', '.jpg', '.jpeg']]

                # Check if any corresponding ground truth file exists
                if not any(os.path.isfile(gt_path) for gt_path in corresponding_gt_paths):
                    create_black_image(corresponding_gt_paths[0])


def print_image_size(image_path):
    # Open the image
    image = Image.open(image_path)

    # Get the size of the image
    width, height = image.size

    # Print the size
    print(f"Image size: {width}x{height} pixels")


def test_result(kittimots_rgb_training_path, kittimots_gt_training_path, dir_name):

    path1 = os.path.join(kittimots_rgb_training_path, dir_name)
    path2 = os.path.join(kittimots_gt_training_path, dir_name)
    output_dir = os.path.join('/Users/leila/Desktop/medvt/dataset/BDD/combined_result/', dir_name)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all files in the directories
    images1 = sorted(os.listdir(path1))
    images2 = sorted(os.listdir(path2))

    # Create a set of filenames for quick lookup
    # images2_set = set(images2)
    images2_set = set(os.path.splitext(img)[0] for img in images2)

    for img1 in images1:
        img1_path = os.path.join(path1, img1)
        img1_name = os.path.splitext(img1)[0]
        img2_path = os.path.join(path2, img1_name + '.png')  # Assuming images have the same names

        # Check if the corresponding image exists in path2
        if img1_name in images2_set:
            # Read the images
            image1 = cv2.imread(img1_path)
            image2 = cv2.imread(img2_path)

            # Check if the images were read correctly
            if image1 is None or image2 is None:
                print(f"Could not read one of the images: {img1_path} or {img2_path}")
                continue

            # Combine images vertically
            combined_image = cv2.vconcat([image1, image2])

            # Save the combined image
            combined_image_name = f"combined_{img1}"
            combined_image_path = os.path.join(output_dir, combined_image_name)
            cv2.imwrite(combined_image_path, combined_image)
            print(f"Saved combined image: {combined_image_path}")
        else:
            print(f"Skipping {img1} as it does not have a corresponding image in {path2}")
    return output_dir


def resize_image(input_path, output_path, size=(1242, 375)):

    try:
        with Image.open(input_path) as img:
            resized_img = img.resize(size, Image.Resampling.LANCZOS)
            resized_img.save(output_path)
            print(f"Image saved to {output_path} with size {size}")
    except Exception as e:
        print(f"An error occurred with file {input_path}: {e}")


def resize_images_in_directory(input_dir, output_dir, size=(1242, 375)):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Check if the file is an image (basic check)
        if os.path.isfile(input_path) and input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            resize_image(input_path, output_path, size)
        else:
            print(f"Skipping non-image file: {filename}")


def convert_to_black_and_white(input_folder):
    # Process each file in the input directory
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            mask_path = os.path.join(input_folder, filename)

            # Open the image file
            img = Image.open(mask_path).convert('RGB')
            # Convert the image to a numpy array
            img_array = np.array(img)

            # Create an array for the black and white mask
            bw_mask = np.zeros(img_array.shape[:2], dtype=np.uint8)

            # Assume the background is the color of the top-left pixel
            background_color = img_array[0, 0]

            # Iterate through each pixel to determine if it's part of the foreground
            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    if not np.array_equal(img_array[i, j], background_color):
                        bw_mask[i, j] = 255

            # Convert the numpy array back to an image
            bw_image = Image.fromarray(bw_mask)

            # Save the black and white image, replacing the original file
            bw_image.save(mask_path)
            print(f"Converted {filename} to black and white and replaced the original image.")


if __name__ == "__main__":

    bdd_rgb_training_path = '/Users/leila/Desktop/medvt/dataset/BDD/ImageSets'
    bdd_gt_path = '/Users/leila/Desktop/medvt/dataset/BDD/annotations/train'


    # check_image_existence(bdd_rgb_training_path, bdd_gt_path)
    # test_result(bdd_rgb_training_path, bdd_gt_path, dir_name='b2e54795-db1f3bad')
    # print_image_size('/Users/leila/Desktop/medvt/dataset/KITTIMOTS/annotations/375p/0017/000060.png')
    print_image_size('/Users/leila/Desktop/medvt/dataset/BDD/Annotations/validation/b1cd1e94-26dd524f/frame0001.png')
    # convert_to_black_and_white("/Users/leila/Desktop/medvt/dataset/BDD/Annotations/validation/b1e1a7b8-65ec7612")

