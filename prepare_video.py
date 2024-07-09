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

# ./dataset/KITTIMOTS/images/training/image_02/0006 ./dataset/KITTIMOTS/annotations/375p/0006 ./combined_train
# ./dataset/KITTIMOTS/images/training/image_02/0006 ./dataset/KITTIMOTS/predict/0006 ./pred_kitti_img/0006

# ./dataset/KITTIMOTS/images/training/image_02/0006 ./dataset/KITTIMOTS/annotations/375p/0006 ./dataset/KITTIMOTS/predict/gt/0006
# ./dataset/KITTIMOTS/images/training/image_02/0006 ./dataset/KITTIMOTS/predict/pre_on_davis/logits/0006 ./dataset/KITTIMOTS/predict/pod/0006
# ./dataset/KITTIMOTS/images/training/image_02/0006 ./dataset/KITTIMOTS/predict/pre_on_kitti/logits/0006 ./dataset/KITTIMOTS/predict/pok/0006

# ./dataset/BDD/JPEGImages/val/b1d7b3ac-0bdb47dc ./dataset/BDD/Annotations/val/b1d7b3ac-0bdb47dc ./dataset/BDD/predict/gt/b1d7b3ac-0bdb47dc
# ./dataset/BDD/JPEGImages/val/b1d7b3ac-0bdb47dc ./dataset/BDD/predict/pobdd/logits/ab1d7b3ac-0bdb47dc ./dataset/BDD/predict/pobdd/ab1d7b3ac-0bdb47dc
# ./dataset/BDD/JPEGImages/val/b4253085-3de357a1 ./dataset/BDD/predict/pod/logits/b4253085-3de357a1 ./dataset/BDD/predict/pod/b4253085-3de357a1
# ./dataset/BDD/JPEGImages/val/b4253085-3de357a1 ./dataset/BDD/predict/pok/logits/b4253085-3de357a1 ./dataset/BDD/predict/pok/b4253085-3de357a1

def main():
    main_dir = sys.argv[1]
    mask_dir = sys.argv[2]
    out_dir = sys.argv[3]
    files = sorted(os.listdir(main_dir))
    masks = sorted(os.listdir(mask_dir))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for i in range(len(files)):
        img = cv2.imread(main_dir + '/' + files[i])
        mask2 = cv2.imread(mask_dir + '/' + files[i].split('.')[0] + '.png', 0)
        mask2[mask2 > 0.5] = 255
        mask2[mask2 < 0.5] = 0
        overlay = create_overlay(img, mask2, [0, 255])
        output = os.path.join(out_dir, files[i])
        cv2.imwrite(output, overlay)
        print(output)


if __name__ == '__main__':
    main()
