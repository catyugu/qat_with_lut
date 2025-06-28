import torch
from torchvision import datasets, transforms
import numpy as np
import os

def save_full_test_set():
    """
    Loads the entire FashionMNIST test set, preprocesses it, and saves
    the images and labels to separate binary files for the C++ application.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'datasets', 'F_MNIST_data')
    print(f"Loading dataset from: {data_path}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    testset = datasets.FashionMNIST(data_path, download=True, train=False, transform=transform)

    # Note: We set batch_size to the full dataset size to process it all at once.
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

    images, labels = next(iter(testloader))

    # Flatten all images and convert to numpy
    images_numpy = images.view(len(testset), -1).numpy().astype(np.float32)
    labels_numpy = labels.numpy().astype(np.int32) # Use int32 for C++ compatibility

    # Define output paths
    images_path = "test_images.bin"
    labels_path = "test_labels.bin"

    # Save the flattened images
    images_numpy.tofile(images_path)
    print(f"\nSaved {len(images_numpy)} images to '{images_path}'")

    # Save the labels
    labels_numpy.tofile(labels_path)
    print(f"Saved {len(labels_numpy)} labels to '{labels_path}'")
    print("\nDataset export complete. You can now run the C++ application.")

if __name__ == '__main__':
    save_full_test_set()