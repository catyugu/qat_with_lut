import torch
from torchvision import datasets, transforms
import numpy as np
import os

def save_full_test_set():
    """
    Loads the entire FashionMNIST test set, preprocesses it (padding), and saves
    the padded images to a separate binary file for the C++ application.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'datasets', 'F_MNIST_data')
    print(f"Loading dataset from: {data_path}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    testset = datasets.FashionMNIST(data_path, download=True, train=False, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

    images, labels = next(iter(testloader))

    # --- 参数从 C++ main.cpp 中获取 ---
    input_dim_original = 784
    input_dim_padded = 960 # 确保与 C++ 代码中的定义一致

    # 1. 展平所有图像并转换为 numpy (float32)
    images_numpy = images.view(len(testset), input_dim_original).numpy().astype(np.float32)
    labels_numpy = labels.numpy().astype(np.int32) # Use int32 for C++ compatibility

    # 2. 对所有图像进行填充
    print(f"Padding images from {input_dim_original} to {input_dim_padded}...")
    padded_images_numpy = np.zeros((len(testset), input_dim_padded), dtype=np.float32)
    padded_images_numpy[:, :input_dim_original] = images_numpy
    print("Padding complete.")

    # 定义输出路径
    padded_images_path = "test_images_padded_f32.bin"
    labels_path = "test_labels.bin"

    # 保存填充后的浮点图像
    padded_images_numpy.tofile(padded_images_path)
    print(f"\nSaved {len(padded_images_numpy)} padded float32 images to '{padded_images_path}' (Shape: {padded_images_numpy.shape})")

    # 保存标签 (保持不变)
    labels_numpy.tofile(labels_path)
    print(f"Saved {len(labels_numpy)} labels to '{labels_path}'")
    print("\nDataset export complete. You can now run the C++ application.")

if __name__ == '__main__':
    save_full_test_set()