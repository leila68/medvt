import numpy as np
import sys
import cv2
import os
from PIL import Image


def PIL2array(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 4)


def create_overlay(img, mask, colors):
    im = Image.fromarray(np.uint8(img))
    im = im.convert('RGBA')

    mask_color = np.zeros((mask.shape[0], mask.shape[1], 3))
    if len(colors) == 3:
        mask_color[mask == colors[1], 0] = 255
        mask_color[mask == colors[1], 1] = 255
        mask_color[mask == colors[2], 0] = 255
    else:
        mask_color[mask == colors[1], 2] = 255

    overlay = Image.fromarray(np.uint8(mask_color))
    overlay = overlay.convert('RGBA')

    im = Image.blend(im, overlay, 0.7)
    blended_arr = PIL2array(im)[:, :, :3]
    img2 = img.copy()
    img2[mask == colors[1], :] = blended_arr[mask == colors[1], :]
    return img2


def predict_overlay():
    main_dir = sys.argv[1]
    mask_dir = sys.argv[2]
    out_dir = sys.argv[3]

    # Recursively find all image files in main_dir
    image_files = []
    for root, _, filenames in os.walk(main_dir):
        for filename in sorted(filenames):
            if filename.endswith('.png'):
                full_img_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_img_path, main_dir)  # relative to main_dir
                image_files.append(rel_path)

    # print(f"[INFO] Total images found: {len(image_files)}")

    for rel_path in image_files:
        img_path = os.path.join(main_dir, rel_path)
        mask_path = os.path.join(mask_dir, rel_path)
        output_path = os.path.join(out_dir, rel_path)

        # Create output subfolders if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Cannot read image: {img_path}")
            continue

        if not os.path.exists(mask_path):
            print(f"[WARNING] Mask not found for: {rel_path}")
            continue

        mask = cv2.imread(mask_path, 0)
        if mask is None:
            print(f"[ERROR] Cannot read mask: {mask_path}")
            continue

        # print(f"[INFO] Processing {rel_path}")
        # print(f"[DEBUG] Image shape: {img.shape}, Mask shape: {mask.shape}")

        # Resize mask if needed
        if mask.shape != img.shape[:2]:
            print(f"[INFO] Resizing mask from {mask.shape} to {img.shape[:2]}")
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Threshold mask
        # print(f"[DEBUG] Mask stats before thresholding: min={np.min(mask)}, max={np.max(mask)}")
        mask = np.where(mask > 0.5, 255, 0).astype(np.uint8)
        # print(f"[DEBUG] Mask stats after thresholding: min={np.min(mask)}, max={np.max(mask)}")

        # Overlay
        overlay = create_overlay(img, mask, [0, 255])  # You must have this function defined
        cv2.imwrite(output_path, overlay)
        print(f"[INFO] Saved overlay: {output_path}")


def predict_overlay_seq16():
    main_dir = sys.argv[1]
    mask_dir = sys.argv[2]
    out_dir = sys.argv[3]

    # Recursively find all image files in main_dir
    image_files = []
    for root, _, filenames in os.walk(main_dir):
        for filename in sorted(filenames):
            if filename.endswith('.png'):
                full_img_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_img_path, main_dir)
                image_files.append(rel_path)

    for rel_path in image_files:
        img_path = os.path.join(main_dir, rel_path)
        mask_path = os.path.join(mask_dir, rel_path)
        output_path = os.path.join(out_dir, rel_path)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Cannot read image: {img_path}")
            continue

        if not os.path.exists(mask_path):
            print(f"[WARNING] Mask not found for: {rel_path}")
            continue

        mask = cv2.imread(mask_path, 0)
        if mask is None:
            print(f"[ERROR] Cannot read mask: {mask_path}")
            continue

        # Crop the mask from top-left corner to match 1224x370
        expected_height, expected_width = 370, 1224
        if mask.shape[0] >= expected_height and mask.shape[1] >= expected_width:
            mask = mask[:expected_height, :expected_width]
        else:
            print(f"[WARNING] Mask too small to crop: {mask.shape}")
            continue

        # Ensure the cropped mask now matches image size
        if mask.shape != img.shape[:2]:
            print(f"[WARNING] Shape mismatch even after cropping: image={img.shape[:2]} mask={mask.shape}")
            continue

        # Threshold mask
        mask = np.where(mask > 0.5, 255, 0).astype(np.uint8)

        # Overlay
        overlay = create_overlay(img, mask, [0, 255])  # red overlay
        cv2.imwrite(output_path, overlay)
        print(f"[INFO] Saved overlay: {output_path}")


def predict_overlay_save_cropped_mask():
    main_dir = sys.argv[1]
    mask_dir = sys.argv[2]
    overlay_out_dir = sys.argv[3]
    cropped_mask_out_dir = sys.argv[4]  # 🆕 New argument for cropped masks

    # Recursively find all image files in main_dir
    image_files = []
    for root, _, filenames in os.walk(main_dir):
        for filename in sorted(filenames):
            if filename.endswith('.png'):
                full_img_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_img_path, main_dir)
                image_files.append(rel_path)

    for rel_path in image_files:
        img_path = os.path.join(main_dir, rel_path)
        mask_path = os.path.join(mask_dir, rel_path)
        overlay_path = os.path.join(overlay_out_dir, rel_path)
        cropped_mask_path = os.path.join(cropped_mask_out_dir, rel_path)  # 🆕

        os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
        os.makedirs(os.path.dirname(cropped_mask_path), exist_ok=True)  # 🆕

        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Cannot read image: {img_path}")
            continue

        if not os.path.exists(mask_path):
            print(f"[WARNING] Mask not found for: {rel_path}")
            continue

        mask = cv2.imread(mask_path, 0)
        if mask is None:
            print(f"[ERROR] Cannot read mask: {mask_path}")
            continue

        # Top-left crop the mask
        expected_height, expected_width = 370, 1224
        if mask.shape[0] >= expected_height and mask.shape[1] >= expected_width:
            mask = mask[:expected_height, :expected_width]
        else:
            print(f"[WARNING] Mask too small to crop: {mask.shape}")
            continue

        if mask.shape != img.shape[:2]:
            print(f"[WARNING] Shape mismatch even after cropping: image={img.shape[:2]} mask={mask.shape}")
            continue

        # Threshold mask
        bin_mask = np.where(mask > 0.5, 255, 0).astype(np.uint8)

        # Save the binary cropped mask for inference
        cv2.imwrite(cropped_mask_path, bin_mask)

        # Overlay
        overlay = create_overlay(img, bin_mask, [0, 255])
        cv2.imwrite(overlay_path, overlay)

        print(f"[INFO] Saved overlay: {overlay_path}")
        print(f"[INFO] Saved cropped mask: {cropped_mask_path}")


def cityscapes_predict_overlay():
    main_dir = sys.argv[1]      # Path to RGB images
    mask_dir = sys.argv[2]      # Path to GT or prediction masks
    out_dir = sys.argv[3]       # Path to save overlays

    # Recursively find all mask files
    mask_files = []
    for root, _, filenames in os.walk(mask_dir):
        for filename in sorted(filenames):
            if filename.endswith('.png'):
                full_mask_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_mask_path, mask_dir)
                mask_files.append(rel_path)

    print(f"[INFO] Found {len(mask_files)} mask files.")

    for rel_path in mask_files:
        mask_path = os.path.join(mask_dir, rel_path)
        img_path = os.path.join(main_dir, rel_path)  # Use same rel_path to find matching image
        output_path = os.path.join(out_dir, rel_path)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Cannot read image: {img_path}")
            continue

        # Read mask
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            print(f"[ERROR] Cannot read mask: {mask_path}")
            continue

        # Resize mask if needed
        if mask.shape != img.shape[:2]:
            print(f"[INFO] Resizing mask from {mask.shape} to {img.shape[:2]}")
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Threshold mask
        mask = np.where(mask > 0.5, 255, 0).astype(np.uint8)

        # Overlay
        overlay = create_overlay(img, mask, [0, 255])  # or [0, 0, 255] for red
        cv2.imwrite(output_path, overlay)
        print(f"[INFO] Saved overlay: {output_path}")


def stack_images_vertically(path1, path2, path3, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    filenames = sorted([f for f in os.listdir(path1) if f.endswith('.png')])

    print(f"Found {len(filenames)} .png files in {path1}")

    for filename in filenames:
        img1_path = os.path.join(path1, filename)
        img2_path = os.path.join(path2, filename)
        img3_path = os.path.join(path3, filename)

        print(f"\nProcessing: {filename}")
        print(f" -> {img1_path}")
        print(f" -> {img2_path}")
        print(f" -> {img3_path}")

        if not (os.path.exists(img1_path) and os.path.exists(img2_path) and os.path.exists(img3_path)):
            print(f" [!] Skipping {filename} (missing in one of the folders)")
            continue

        try:
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
            img3 = Image.open(img3_path)
        except Exception as e:
            print(f" [!] Failed to open one of the images: {e}")
            continue

        width = min(img1.width, img2.width, img3.width)
        img1 = img1.resize((width, int(img1.height * width / img1.width)))
        img2 = img2.resize((width, int(img2.height * width / img2.width)))
        img3 = img3.resize((width, int(img3.height * width / img3.width)))

        total_height = img1.height + img2.height + img3.height
        stacked_img = Image.new('RGB', (width, total_height))
        stacked_img.paste(img1, (0, 0))
        stacked_img.paste(img2, (0, img1.height))
        stacked_img.paste(img3, (0, img1.height + img2.height))

        output_path = os.path.join(output_dir, filename)
        stacked_img.save(output_path)
        print(f" [✓] Saved to {output_path}")

    print("\nAll done!")

# ./dataset/Cityscapes/leftImg8bit_sequence/lindau ./dataset/Cityscapes/Annotations/val/lindau ./dataset/Cityscapes/prediction_overlay/gt/lindau
# ./dataset/Cityscapes/leftImg8bit_sequence/lindau ./dataset/Cityscapes/prediction_overlay/msq/logits-msq/lindau ./dataset/Cityscapes/prediction_overlay/msq/lindau
# ./dataset/Cityscapes/leftImg8bit_sequence/lindau ./dataset/Cityscapes/prediction_overlay/msqm/logits-msqm/lindau ./dataset/Cityscapes/prediction_overlay/msqm/lindau

# ./dataset/Cityscapes/leftImg8bit_sequence/munster ./dataset/Cityscapes/Annotations/val/munster ./dataset/Cityscapes/prediction_overlay/gt/munster
# ./dataset/Cityscapes/leftImg8bit_sequence/munster ./dataset/Cityscapes/prediction_overlay/msq/logits-msq/munster ./dataset/Cityscapes/prediction_overlay/msq/munster
# ./dataset/Cityscapes/leftImg8bit_sequence/munster ./dataset/Cityscapes/prediction_overlay/msqm/logits-msqm/munster ./dataset/Cityscapes/prediction_overlay/msqm/munster


# ./dataset/KITTIMOTS/images/training/image_02/0016 ./dataset/KITTIMOTS/annotations/375p/0016 ./dataset/KITTIMOTS/predicted_overlay/gt/0016_crop ./dataset/KITTIMOTS/annotations/375p/0016_crop

# ./dataset/KITTIMOTS/images/training/image_02/0016 ./dataset/KITTIMOTS/annotations/375p/0016_crop ./dataset/KITTIMOTS/predicted_overlay/gt/0016_crop


if __name__ == '__main__':

    predict_overlay()
    # predict_overlay_seq16()
    # predict_overlay_save_cropped_mask()
    # cityscapes_predict_overlay()


    # gt, predictions in one for comparison
    # stack_images_vertically("/Users/leila/Desktop/medvt/dataset/Cityscapes/cityscapes_prediction_result/gt/frankfurt", "/Users/leila/Desktop/medvt/dataset/Cityscapes/cityscapes_prediction_result/msq/frankfurt",
    #                         "/Users/leila/Desktop/medvt/dataset/Cityscapes/cityscapes_prediction_result/msqm/frankfurt", "/Users/leila/Desktop/medvt/dataset/Cityscapes/cityscapes_prediction_result/compare_result/frankfurt")


