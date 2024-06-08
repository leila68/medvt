import cv2
import torch
import numpy as np
import mmcv
import matplotlib.pyplot as plt
from avos.datasets.train.davis16_train_data import Davis16TrainDataset
from avos.datasets.train.kittimots_train_data import KittimotsTrainDataset
from avos.datasets.test.kittimots_val_data import KittimotsValDataset
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

    # plt.show()
    # plt.close()


if __name__ == "__main__":

    obj_test = KittimotsTrainDataset(num_frames=1, train_size=300, use_ytvos=True, use_flow=True)
    # obj_test = KittimotsValDataset(num_frames=1, val_size=300,  use_flow=False)

    #for obj in obj_test:
    for idx, obj in enumerate(obj_test):
        # print(obj[0].data)
        denormalized_image = denormalize(obj[0].data, mean=(0.485, 0.456, 0), std=(0.229, 0.224, 0.225))
        plt.imshow(denormalized_image)
        plt.axis('off')  # Turn off axis labels
        plt.savefig("kitti_images_%05d.png"%idx, bbox_inches='tight', pad_inches=0)

        # kittimots mask
        numpy_img_mask = obj[1]['masks'].permute(1, 2, 0).numpy()
        numpy_img_mask = (numpy_img_mask * 255).astype('uint8')
        cv2.imwrite('kitti_images_mask_%05d.png'%idx, numpy_img_mask)

        # exit(0)







