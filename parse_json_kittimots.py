from pycocotools.coco import COCO
import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def create_image_path(kittimots_training_path, kittimots_train_json_file, id):
  with open(kittimots_train_json_file, 'r') as f:
    data = json.load(f)

  for entry in data['annotations']:
    if entry['id'] == id:
       file_name = entry['file_name']
       file_path = os.path.join(kittimots_training_path, file_name)
       print(file_path)
       plt.imshow(mpimg.imread(file_path))
       plt.show()
    else:
      print("Skipping image {}".format(entry['id']))


def image_path(kittimots_path,file_name):
    # print('path', os.path.join(kittimots_path, file_name))
    return os.path.join(kittimots_path, file_name)


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


def subtract_masks(mask1, mask2):
    # Perform pixel-wise subtraction
    difference_mask = np.abs(mask2 - mask1)

    # Check if all elements of the mask are zero
    if np.all(difference_mask == 0):
        print("There is no motion from first frame to second frame")
    else:
        print("There is motion from  first frame to second frame")

    # Display the result
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.title('Mask 1')
    plt.imshow(mask1,  interpolation='nearest', cmap='grey',)
    # plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title('Mask 2')
    plt.imshow(mask2, cmap='grey', interpolation='nearest')
    # plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title('Difference Mask')
    plt.imshow(difference_mask, cmap='grey', interpolation='nearest')
    # plt.colorbar()

    # Save the combined image as a PNG file
    plt.savefig('diff_mask.png', bbox_inches='tight', pad_inches=0)

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
  kittimots_ann_train_path = '/Users/leila/Desktop/medvt/dataset/KITTIMOTS/annotations/training/'
  kittimots_ann_val_path = '/Users/leila/Desktop/medvt/dataset/KITTIMOTS/annotations/validation/'
  kittimots_training_path = '/Users/leila/Desktop/medvt/dataset/KITTIMOTS/images/training/image_02'
  kittimots_train_json_file = '/Users/leila/Desktop/medvt/dataset/KITTIMOTS/annotations/KITTIMOTS_MOSeg_train.json'
  kittimots_val_json_file = '/Users/leila/Desktop/medvt/dataset/KITTIMOTS/annotations/KITTIMOTS_MOSeg_val.json'
  kittimots_val_fix_json_file = '/Users/leila/Desktop/medvt/dataset/KITTIMOTS/annotations/KITTIMOTS_MOSeg_val.json'

  # img_show = mpimg.imread('/Users/leila/Desktop/medvt/dataset/KITTIMOTS/annotations/training/0011/000051.png')
  # img_mks_show = mpimg.imread('/Users/leila/Desktop/medvt/dataset/KITTIMOTS/images/training/image_02/0011/000051.png')
  # show_images_with_mask(img_mks_show, img_show)

  coco_ds = COCO(kittimots_train_json_file)

  ann_id = coco_ds.getAnnIds(imgIds=[111])
  ann = coco_ds.loadAnns(ann_id)
  mask1 = coco_ds.annToMask(ann[0])

  ann_id = coco_ds.getAnnIds(imgIds=[112])
  ann = coco_ds.loadAnns(ann_id)
  mask2 = coco_ds.annToMask(ann[0])

  # subtract_masks(mask1, mask2)


  img_ids = coco_ds.getImgIds()
  for img_id in img_ids:
      ann_id = coco_ds.getAnnIds(imgIds=[img_id])
      ann = coco_ds.loadAnns(ann_id)
      mask = coco_ds.annToMask(ann[0])
      img_load = coco_ds.loadImgs(img_id)[0]
      img_path = image_path(kittimots_ann_train_path, img_load['file_name'])
      # plt.imshow(mask, cmap='gray')
      # plt.axis('off')
      # plt.show()

      save_mask_images(kittimots_ann_train_path, mask, img_load['file_name'])
      img = mpimg.imread(img_path)
      img = mpimg.imread(img_path)
      # show_images_with_mask(img, mask)
      # exit()

