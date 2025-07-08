import os
import torch
import argparse
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.utils as tv_utils
from PIL import Image

# Import the necessary components from your project structure
from ddpm.QAT_UNet import QATUNet
from ddpm.diffusion import GaussianDiffusion, generate_linear_schedule
from ddpm.script_utils_qat import get_transform, cycle

parser = argparse.ArgumentParser(description="Robust QAT Training Script")
parser.add_argument("--dataset_path", type=str, default="./data", help="Path to CIFAR-10 dataset")
parser.add_argument("--save_path", type=str, default="./qat_unet_final.pth", help="Path to save the trained model")
parser.add_argument("--device", type=str, default="cuda:3" if torch.cuda.is_available() else "cpu")
# Diffusion Hyperparameters
parser.add_argument("--num_timesteps", type=int, default=1000)

# UNet Hyperparameters
parser.add_argument("--base_channels", type=int, default=128)
parser.add_argument("--channel_mults", type=str, default="1,2,2,2")
parser.add_argument("--num_res_blocks", type=int, default=2)
parser.add_argument("--time_emb_dim", type=int, default=512)
parser.add_argument("--norm", type=str, default="gn")
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--attention_resolutions", type=str, default="1", help="Resolutions for attention blocks")

args = parser.parse_args()

args.channel_mults = tuple(map(int, args.channel_mults.split(',')))
args.attention_resolutions = tuple(map(int, args.attention_resolutions.split(',')))
model = QATUNet(
    img_channels=3, base_channels=args.base_channels, channel_mults=args.channel_mults,
    num_res_blocks=args.num_res_blocks, time_emb_dim=args.time_emb_dim, time_emb_scale=1.0,
    num_classes=10, dropout=args.dropout, attention_resolutions=args.attention_resolutions,
    norm=args.norm, num_groups=32, initial_pad=0,
)
model.load_state_dict(torch.load('qat_unet_final.pth')) # Load your exact weights
model.eval()


# 2. Create a simple, reproducible input tensor
#    Using torch.ones makes it easy to create the same tensor in C++
input_tensor = torch.ones(1, 3, 32, 128, dtype=torch.float32)

# 3. Perform ONLY the initial convolution
with torch.no_grad():
    # Get the initial convolution layer
    init_conv_layer = model.init_conv
    # Pass the input through it
    output_tensor = init_conv_layer(input_tensor)


# 4. Print the golden standard statistics
print("--- Python Golden Trace (after init_conv) ---")
print(f"Output Shape: {output_tensor.shape}")
print(f"Output Mean: {output_tensor.mean().item():.8f}")
print(f"Output Std Dev: {output_tensor.std().item():.8f}")
print(f"Sum of all elements: {output_tensor.sum().item():.8f}")
print("-" * 20)
# Print specific, hard-to-fake values
print(f"Value at [0, 0, 0, 0]: {output_tensor[0, 0, 0, 0].item():.8f}")
print(f"Value at [0, 5, 10, 15]: {output_tensor[0, 5, 10, 15].item():.8f}")
# Crucially, check the channel means
for i in range(min(5, 128)): # Print first 5 channel means
    print(f"Mean of Channel {i}: {output_tensor[:, i, :, :].mean().item():.8f}")