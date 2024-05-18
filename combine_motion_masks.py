from pycocotools.coco import COCO
import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt
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
    width = 1242
    height = 375
    # Create a black image (all zeros)
    black_image = Image.new('RGB', (width, height), color=(0, 0, 0))

    # Save the image
    black_image.save(path)
    print(f"Black image saved to {path}")


def check_image_existence(kittimots_rgb_training_path, kittimots_gt_training_path):

    image_paths = []
    # Iterate through folders in the base path
    for folder_index, folder_name in enumerate(sorted(os.listdir(kittimots_rgb_training_path))):
        folder_path = os.path.join(kittimots_rgb_training_path, folder_name)

        if os.path.isdir(folder_path):
            # Iterate through images in the current folder
            for image_index, image_name in enumerate(sorted(os.listdir(folder_path))):
                image_path = os.path.join(folder_path, image_name)
                gt_path = os.path.join(kittimots_gt_training_path, folder_name)
                if os.path.exists(gt_path):
                    gt_path = os.path.join(kittimots_gt_training_path, folder_name, image_name)
                    if (os.path.isfile(gt_path)):
                        continue
                    else:
                        create_black_image(gt_path)


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
    output_dir = os.path.join('/Users/leila/Desktop/medvt/dataset/KITTIMOTS/test_result/', dir_name)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all files in the directories
    images1 = sorted(os.listdir(path1))
    images2 = sorted(os.listdir(path2))

    # Create a set of filenames for quick lookup
    images2_set = set(images2)

    for img1 in images1:
        img1_path = os.path.join(path1, img1)
        img2_path = os.path.join(path2, img1)  # Assuming images have the same names

        # Check if the corresponding image exists in path2
        if img1 in images2_set:
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


def create_gt_files(kittimots_train_json_file, kittimots_gt_training_path):
    coco_ds = COCO(kittimots_train_json_file)
    img_ids = coco_ds.getImgIds()

    for img_id in img_ids:
        ann_id = coco_ds.getAnnIds(imgIds=[img_id])
        all_masks = []
        counter = 0
        for ann in ann_id:
            ann = coco_ds.loadAnns(ann)
            if ann[0]['category_id'] == 0:
                mask = coco_ds.annToMask(ann[0])
                all_masks.append(mask)
                counter += 1

        if len(all_masks) == 0:
            continue
        combined_mask = np.logical_or.reduce(all_masks)
        print("motion of ", img_id, ":", counter)
        img_load = coco_ds.loadImgs(img_id)[0]
        img_path = image_path(kittimots_gt_training_path, img_load['file_name'])
        save_mask_images(kittimots_gt_training_path, combined_mask, img_load['file_name'])

        # plt.imshow(combined_mask, cmap='gray')
        # plt.title(img_id)
        # plt.axis('off')
        # plt.show()
        # plt.close()
        # exit()


if __name__ == "__main__":
    kittimots_train_json_file = '/Users/leila/Desktop/medvt/dataset/KITTIMOTS/annotations/KITTIMOTS_MOSeg_train.json'
    kittimots_gt_training_path = '/Users/leila/Desktop/medvt/dataset/KITTIMOTS/annotations/training/'
    kittimots_rgb_training_path = '/Users/leila/Desktop/medvt/dataset/KITTIMOTS/images/training/image_02'

    # create_gt_files(kittimots_train_json_file, kittimots_gt_training_path)
    # check_image_existence(kittimots_rgb_training_path, kittimots_gt_training_path)
    # test_result(kittimots_rgb_training_path, kittimots_gt_training_path, dir_name='0014')
    # print_image_size('/Users/leila/Desktop/medvt/dataset/KITTIMOTS/annotations/motion/0000/000000.png')

