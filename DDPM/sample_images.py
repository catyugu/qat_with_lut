# DDPM/train_qat_cifar.py
# Final robust training script using Progressive Quantization.

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

def main():
    parser = argparse.ArgumentParser(description="Robust QAT Training Script")

    # Paths
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

    print("="*40)
    print("Starting Sampling with Quantized UNet")
    print("="*40)
    model = QATUNet(
        img_channels=3, base_channels=args.base_channels, channel_mults=args.channel_mults,
        num_res_blocks=args.num_res_blocks, time_emb_dim=args.time_emb_dim, time_emb_scale=1.0,
        num_classes=10, dropout=args.dropout, attention_resolutions=args.attention_resolutions,
        norm=args.norm, num_groups=32, initial_pad=0,
    ).to(args.device)

    # --- SETUP ---
    model.set_quantize_weights(True)
    model.set_quantize_activations(True)
    betas = generate_linear_schedule(args.num_timesteps, low=1e-4, high=0.02)
    diffusion = GaussianDiffusion(model, (32, 32), 3, 10, betas=betas, loss_type="l2").to(args.device)
    
    ## load pre-trained weights if available
    if os.path.exists(args.save_path):
        print(f"Loading pre-trained model from {args.save_path}")
        model.load_state_dict(torch.load(args.save_path, map_location=args.device))


    diffusion.ema_model.load_state_dict(model.state_dict())
    labels_to_sample = torch.arange(10, device=args.device)
    model.eval() 
    samples = diffusion.sample(batch_size=10, device=args.device, y=labels_to_sample, use_ema=True)
    tv_utils.save_image(
        samples,
        "samples.png",
        nrow=10,
        normalize=True,
        value_range=(-1, 1),
    )
    print("Sample images saved to 'samples.png'")

if __name__ == "__main__":
    main()