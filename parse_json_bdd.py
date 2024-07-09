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


def save_mask_images(bdd_val_path, mask_data, file_name):

    os.makedirs(bdd_val_path, exist_ok=True)

    if file_name.lower().endswith('.jpg'):
        file_name = file_name[:-4] + '.png'

    # Save the mask image with the same name in the corresponding directory
    mask_file_path = os.path.join(bdd_val_path, file_name)

    # Convert mask data to numpy array
    mask_array = np.array(mask_data, dtype=np.uint8)

    # Scale the mask to range 0-255
    mask_scaled = mask_array * 255

    # Save the mask image as PNG
    cv2.imwrite(mask_file_path, mask_scaled)

    print("image is saved in:", mask_file_path)

def combine_mask_files(path1, path2, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all mask files in the first directory
    mask_files = [f for f in os.listdir(path1) if os.path.isfile(os.path.join(path1, f))]

    for file_name in mask_files:
        # Construct the full file paths for the masks in both directories
        mask_path1 = os.path.join(path1, file_name)
        mask_path2 = os.path.join(path2, file_name)
        output_file_path = os.path.join(output_dir, file_name)

        # Check if the corresponding mask file exists in the second directory
        if os.path.exists(mask_path2):
            # Read the mask images
            mask1 = cv2.imread(mask_path1, cv2.IMREAD_GRAYSCALE)
            mask2 = cv2.imread(mask_path2, cv2.IMREAD_GRAYSCALE)

            if mask1 is None or mask2 is None:
                print(f"Error reading masks for {file_name}")
                continue

            # Combine the masks using a logical OR operation
            combined_mask = np.maximum(mask1, mask2)

            # Save the combined mask as PNG
            cv2.imwrite(output_file_path, combined_mask)
            print(f"Combined mask saved as: {output_file_path}")
        else:
            print(f"Mask file {file_name} not found in {path2}")


if __name__ == "__main__":

  # bdd_ann_val_path = '/Users/leila/Desktop/medvt/dataset/BDD/Annotations/my_annotation/b1e1a7b8-65ec7612'
  # bdd_val_path = '/Users/leila/Desktop/medvt/dataset/BDD/JPEGImages/val/b1e1a7b8-65ec7612'
  # bdd_seq_json_file = '/Users/leila/Desktop/medvt/my_annotation_json/b1e1a7b8-65ec7612_coco.json'
  #
  # combine_mask_files('/Users/leila/Desktop/medvt/dataset/BDD/Annotations/val/b1e1a7b8-65ec7612',
  #                    '/Users/leila/Desktop/medvt/dataset/BDD/Annotations/my_annotation/b1e1a7b8-65ec7612',
  #                    '/Users/leila/Desktop/medvt/dataset/BDD/Annotations/my_annotation/out_612')

  bdd_ann_val_path = '/Users/leila/Desktop/medvt/dataset/BDD/Annotations/my_annotation/7dc_dog'
  bdd_val_path = '/Users/leila/Desktop/medvt/dataset/BDD/JPEGImages/val/b1d7b3ac-0bdb47dc'
  bdd_seq_json_file = '/Users/leila/Desktop/medvt/my_annotation_json/7dc_dog_coco.json'

  combine_mask_files('/Users/leila/Desktop/medvt/dataset/BDD/Annotations/my_annotation/out_7dc',
                     '/Users/leila/Desktop/medvt/dataset/BDD/Annotations/my_annotation/7dc_dog',
                     '/Users/leila/Desktop/medvt/dataset/BDD/Annotations/my_annotation/out_7dc_dog')

  coco_ds = COCO(bdd_seq_json_file)

  # ann_id = coco_ds.getAnnIds(imgIds=[4])
  # ann = coco_ds.loadAnns(ann_id)
  # mask1 = coco_ds.annToMask(ann[0])
  # plt.imshow(mask1, cmap='gray')
  # plt.axis('off')
  # plt.show()
  #
  # img_ids = coco_ds.getImgIds()
  # for img_id in img_ids:
  #     ann_id = coco_ds.getAnnIds(imgIds=[img_id])
  #     all_masks = []
  #     for ann in ann_id:
  #         ann = coco_ds.loadAnns(ann)
  #         mask = coco_ds.annToMask(ann[0])
  #         all_masks.append(mask)
  #
  #     combined_mask = np.logical_or.reduce(all_masks)
  #     img_load = coco_ds.loadImgs(img_id)[0]
  #     # img_name = image_path(kittimots_ann_train_path, img_load['file_name'])
  #     save_mask_images(bdd_ann_val_path, combined_mask, img_load['file_name'])
  #
  #     # plt.imshow(combined_mask, cmap='gray')
  #     # plt.title(img_id)
  #     # plt.axis('off')
  #     # plt.show()
  #     # plt.close()
  #     print('print')
  #     # exit()

