import cv2
import torch
import numpy as np
from PIL import Image
import mmcv
import matplotlib.pyplot as plt
from avos.datasets.train.bdd_train_data import BddTrainDataset
from avos.datasets.test.bdd_val_data import BddValDataset
import os


def denormalize(tensor, mean=(0, 0, 0), std=(1, 1, 1)):
    # Mean and standard deviation used for normalization
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    # Reshape mean and std to match the dimensions of the tensor
    mean = mean.view(3, 1, 1)
    std = std.view(3, 1, 1)
    # Denormalize the tensor
    denormalized_tensor = (obj[0].data * std) + mean
    # Convert the tensor to a NumPy array
    denormalized_image = denormalized_tensor.numpy()
    # Clip values to the range [0, 1]
    denormalized_image = np.clip(denormalized_image, 0, 1)
    # Transpose the image to HWC format for matplotlib
    denormalized_image = np.transpose(denormalized_image, (1, 2, 0))
    return denormalized_image

    # plt.imshow(denormalized_image)
    # plt.axis('off')  # Turn off axis labels
    # plt.savefig("bdd_image.png", bbox_inches='tight', pad_inches=0)
    # plt.show()
    # plt.close()


def print_image_size(image_path):
    # Open the image
    image = Image.open(image_path)

    # Get the size of the image
    width, height = image.size

    # Print the size
    print(f"Image size: {width}x{height} pixels")


if __name__ == "__main__":
    obj_test = BddValDataset(num_frames=1, val_size=720, use_flow=False)

    print_image_size('/Users/leila/Desktop/medvt/bdd_images_00002.png')
    print_image_size('/Users/leila/Desktop/medvt/bdd_images_mask_00002.png')
    print_image_size('/Users/leila/Desktop/medvt/dataset/BDD/JPEGImages/720p/b1cd1e94-26dd524f/frame0001.jpg')
    print_image_size('/Users/leila/Desktop/medvt/dataset/BDD/Annotations/720p/b1cd1e94-26dd524f/frame0001.png')

    for idx, obj in enumerate(obj_test):
        # print(obj[0].data)
        denormalized_image = denormalize(obj[0].data, mean=(0.485, 0.456, 0), std=(0.229, 0.224, 0.225))
        plt.imshow(denormalized_image)
        plt.axis('off')  # Turn off axis labels
        plt.savefig("bdd_images_%05d.png" % idx, bbox_inches='tight', pad_inches=0)

        # davis mask
        numpy_img_mask = obj[1]['masks'].permute(1, 2, 0).numpy()
        numpy_img_mask = (numpy_img_mask * 255).astype('uint8')
        cv2.imwrite('bdd_images_mask_%05d.png'%idx, numpy_img_mask)
        print(idx)

        exit(0)








