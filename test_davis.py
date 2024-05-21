import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from avos.datasets.train.davis16_train_data import Davis16TrainDataset


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

    plt.imshow(denormalized_image)
    plt.axis('off')  # Turn off axis labels
    plt.savefig("davis_image.png", bbox_inches='tight', pad_inches=0)
    # plt.show()
    # plt.close()


if __name__ == "__main__":
    obj_test = Davis16TrainDataset(num_frames=1, train_size=300, use_ytvos=True, use_flow=True)

    for obj in obj_test:
        # print(obj[0].data)
        denormalize(obj[0].data, mean=(0.485, 0.456, 0), std=(0.229, 0.224, 0.225))

        # davis mask
        numpy_img_mask = obj[1]['masks'].permute(1, 2, 0).numpy()
        numpy_img_mask = (numpy_img_mask * 255).astype('uint8')
        cv2.imwrite('davis_image_mask.png', numpy_img_mask)

        exit(0)







