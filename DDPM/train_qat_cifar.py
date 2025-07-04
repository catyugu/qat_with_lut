# DDPM/train_qat_progressive.py
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
    parser.add_argument("--save_path", type=str, default="./qat_unet_progressive.pth", help="Path to save the trained model")
    parser.add_argument("--sample_dir", type=str, default="./samples", help="Directory to save generated samples")
    # Training Hyperparameters
    parser.add_argument("--device", type=str, default="cuda:2" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--total_train_steps", type=int, default=60000, help="Total steps.")
    parser.add_argument("--sample_interval", type=int, default=1000, help="Frequency of saving sample images.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    
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
    print("Starting Progressive Quantization Training")
    print("="*40)

    dataset = torchvision.datasets.CIFAR10(
        root=args.dataset_path, train=True, download=True, transform=get_transform()
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    model = QATUNet(
        img_channels=3, base_channels=args.base_channels, channel_mults=args.channel_mults,
        num_res_blocks=args.num_res_blocks, time_emb_dim=args.time_emb_dim, time_emb_scale=1.0,
        num_classes=None, dropout=args.dropout, attention_resolutions=args.attention_resolutions,
        norm=args.norm, num_groups=32, initial_pad=0,
    ).to(args.device)

    # --- SETUP ---
    model.set_quantize_weights(True)
    model.set_quantize_activations(True)
    betas = generate_linear_schedule(args.num_timesteps, low=1e-4, high=0.02)
    diffusion = GaussianDiffusion(model, (32, 32), 3, betas=betas, loss_type="l2",num_classes=10).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.total_train_steps)

    pbar = tqdm(range(args.total_train_steps))
    data_iter = cycle(dataloader)
    
    for i in pbar:

        optimizer.zero_grad()
        x, y = next(data_iter)
        x = x.to(args.device)
        y = y.to(args.device)
        
        loss = diffusion(x,y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        diffusion.update_ema()
        pbar.set_description(f"Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.1e}")
        if (i + 1) % args.sample_interval == 0:
            model.eval() # Set model to evaluation mode
            with torch.no_grad():
                labels_to_sample = torch.arange(10, device=args.device)
                samples = diffusion.sample(batch_size=10, device=args.device, y=labels_to_sample, use_ema=True)
                tv_utils.save_image(
                    samples,
                    os.path.join(args.sample_dir, f"sample_step_{i+1}.png"),
                    nrow=10,
                    normalize=True,
                    value_range=(-1, 1),
                )
            model.train() # Set model back to training mode
        if (i + 1) % 1000 == 0:
            torch.save(model.state_dict(), args.save_path)
            print(f"\nSaved model checkpoint at step {i+1} to {args.save_path}")

    print("Training complete.")
    torch.save(model.state_dict(), args.save_path)
    print(f"Final model saved to {args.save_path}")

if __name__ == "__main__":
    main()
