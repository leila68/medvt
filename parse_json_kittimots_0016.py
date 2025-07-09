from pycocotools.coco import COCO
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def create_overlay_result(image_dir, mask_dir, out_dir, sequence='0016'):
    for root, _, files in os.walk(os.path.join(mask_dir, sequence)):
        for file in sorted(files):
            if not file.endswith('.png'):
                continue

            relative_path = os.path.relpath(root, mask_dir)
            img_path = os.path.join(image_dir, relative_path, file)
            mask_path = os.path.join(root, file)

            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue

            img = cv2.imread(img_path)         # BGR
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img.shape[:2] != mask.shape:
                print(f"Shape mismatch: {file} -> image {img.shape[:2]}, mask {mask.shape}")
                continue

            # print(f"[DEBUG] {file}")
            # print(f"Image shape: {img.shape}")
            # print(f"Mask shape: {mask.shape}")

            # Overlay (red mask)
            overlay = overlay_mask_on_image(img, mask, alpha=0.5, mask_color=(0, 0, 255))  # Red in BGR

            out_seq_dir = os.path.join(out_dir, relative_path)
            os.makedirs(out_seq_dir, exist_ok=True)
            out_path = os.path.join(out_seq_dir, file)
            cv2.imwrite(out_path, overlay)
            print(f"Overlay saved: {out_path}")


def image_path(base_path, file_name):
    return os.path.join(base_path, file_name)


def overlay_mask_on_image(image, mask, alpha=0.5, mask_color=(0, 0, 255)):
    overlay = image.copy()
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = mask_color

    # Apply transparency
    mask_indices = mask > 0
    overlay[mask_indices] = cv2.addWeighted(
        image[mask_indices], 1 - alpha, colored_mask[mask_indices], alpha, 0
    )

    # Draw green contours for debugging
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)

    return overlay


def resize_images(input_dir, output_dir, target_size=(1224, 370)):
    """
    Resize all images in a directory to the given target size and save them to the output directory.

    Parameters:
    - input_dir (str): Path to the input directory containing image files.
    - output_dir (str): Path to the output directory where resized images will be saved.
    - target_size (tuple): Desired size as (width, height). Default is (1224, 370).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        # Skip non-image files
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            continue

        img = cv2.imread(input_path)
        if img is None:
            print(f"Warning: Unable to read {input_path}")
            continue

        resized_img = cv2.resize(img, target_size)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, resized_img)

    print(f"All images resized and saved to {output_dir}")


if __name__ == "__main__":

    kittimots_val_json_file = '/Users/leila/Desktop/medvt/dataset/KITTIMOTS/annotations/KITTIMOTS_MOSeg_val_fix.json'
    kittimots_val_img_path = '/Users/leila/Desktop/medvt/dataset/KITTIMOTS/images/training/image_02'
    kittimots_val_mask_out = '/Users/leila/Desktop/medvt/dataset/KITTIMOTS/annotations/validation_fix/'
    overlay_out_dir = '/Users/leila/Desktop/medvt/dataset/KITTIMOTS/overlay_results/'

    # parse_json(kittimots_val_json_file, kittimots_val_img_path, kittimots_val_mask_out)

    image_path = "/Users/leila/Desktop/medvt/dataset/KITTIMOTS/images/training/image_02/0016/000010.png"
    image = cv2.imread(image_path)

    mask_path = "/Users/leila/Desktop/medvt/dataset/KITTIMOTS/annotations/validation_fix/0016/000010.png"
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # right,left -- up,down
    M = np.float32([[1, 0, 11], [0, 1, 0]])
    mask_shifted = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

    overlay = overlay_mask_on_image(image, mask_shifted, alpha=0.5, mask_color=(0, 0, 255))

    cv2.imwrite("overlay_result_10_11.png", overlay)  # Save to file

    # resize_images("/Users/leila/Desktop/medvt/dataset/KITTIMOTS/images/training/image_02/0016",
    #       "/Users/leila/Desktop/medvt/dataset/KITTIMOTS/images/training/image_02/0016_resize")

    # Call overlay creation after saving all masks
    # create_overlay_result(
    #     image_dir=kittimots_val_img_path,
    #     mask_dir=kittimots_val_mask_out,
    #     out_dir=overlay_out_dir,
    #     sequence='0016'
    # )
