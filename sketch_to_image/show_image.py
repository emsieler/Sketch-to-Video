import matplotlib.pyplot as plt
import torchvision
import torch

def show_image(tensor_image, title=None):
    tensor_image = tensor_image * 0.5 + 0.5
    np_image = tensor_image.cpu().detach().numpy().transpose(1, 2, 0)
    plt.imshow(np_image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
