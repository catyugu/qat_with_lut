import torch
from torchvision import datasets, transforms
import numpy as np
import os

def save_single_image():
    """
    Loads one image from the FashionMNIST test set, preprocesses it,
    and saves it to a simple binary file for the C++ application to read.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'datasets', 'F_MNIST_data')

    # IMPORTANT: The transform must be identical to the one used in training
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load the test set
    testset = datasets.FashionMNIST(data_path, download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    # Get a single image and its label
    image, label = next(iter(testloader))

    # The true class of the image we saved
    # Item     | Label
    #-----------------
    # T-shirt  | 0
    # Trouser  | 1
    # Pullover | 2
    # Dress    | 3
    # Coat     | 4
    # Sandal   | 5
    # Shirt    | 6
    # Sneaker  | 7
    # Bag      | 8
    # Ankle boot| 9
    print(f"Saving one test image. The correct label is: {label.item()}")

    # Flatten the image tensor and convert to a numpy array
    image_numpy = image.view(1, -1).numpy().astype(np.float32)

    # Save to a simple binary file
    output_path = "test_image.bin"
    image_numpy.tofile(output_path)

    print(f"Image saved to '{output_path}'. It contains {image_numpy.size} float values.")

if __name__ == '__main__':
    save_single_image()