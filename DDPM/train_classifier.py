# DDPM/train_classifier.py
# A script to train a classifier on top of the frozen features of a pre-trained QAT-UNet.

import os
import torch
import argparse
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

# Import the necessary components from your project structure
from ddpm.QAT_UNet import QATUNet
from ddpm.script_utils_qat import get_transform, cycle

class FeatureClassifier(nn.Module):
    """A simple classifier to be trained on top of the UNet's features."""
    def __init__(self, feature_dim, num_classes=10):
        super().__init__()
        # The UNet's middle block output is (batch_size, channels, 4, 4)
        # We flatten this to (batch_size, channels * 4 * 4)
        self.flatten = nn.Flatten()
        self.classifier_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, features):
        x = self.flatten(features)
        return self.classifier_head(x)

def main():
    parser = argparse.ArgumentParser(description="Train a classifier on frozen UNet features.")
    
    # Paths
    parser.add_argument("--dataset_path", type=str, default="./data", help="Path to CIFAR-10 dataset")
    parser.add_argument("--qat_model_path", type=str, default="./qat_unet_progressive.pth", help="Path to the pre-trained QAT UNet model.")
    
    # Training Hyperparameters
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    
    # UNet Hyperparameters (must match the trained model)
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
    print("Starting Classifier Training on QAT-UNet Features")
    print("="*40)

    # --- 1. Load Datasets ---
    train_dataset = torchvision.datasets.CIFAR10(
        root=args.dataset_path, train=True, download=True, transform=get_transform()
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.dataset_path, train=False, download=True, transform=get_transform()
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- 2. Load and Freeze the QAT-UNet ---
    feature_extractor = QATUNet(
        img_channels=3, base_channels=args.base_channels, channel_mults=args.channel_mults,
        num_res_blocks=args.num_res_blocks, time_emb_dim=args.time_emb_dim, time_emb_scale=1.0,
        num_classes=None, dropout=args.dropout, attention_resolutions=args.attention_resolutions,
        norm=args.norm, num_groups=32, initial_pad=0,
    ).to(args.device)
    
    print(f"Loading pre-trained QAT-UNet from: {args.qat_model_path}")
    feature_extractor.load_state_dict(torch.load(args.qat_model_path, map_location=args.device))
    
    # Freeze the feature extractor
    for param in feature_extractor.parameters():
        param.requires_grad = False
    feature_extractor.eval()
    print("QAT-UNet loaded and frozen.")

    # --- 3. Initialize the Classifier ---
    # Determine feature dimension from a dummy pass
    dummy_input = torch.randn(1, 3, 32, 32).to(args.device)
    dummy_time = torch.zeros(1, dtype=torch.long).to(args.device)
    
    # We will extract features from the middle of the network
    # To do this, we modify the forward method slightly to return the mid-block output
    def get_features(self, x, time=None, y=None):
        time_emb = self.time_mlp(time) if self.time_mlp is not None else None
        x = self.init_conv(x)
        skips = [x]
        for layer in self.downs:
            x = layer(x, time_emb, y)
            skips.append(x)
        for layer in self.mid:
            x = layer(x, time_emb, y)
        return x # Return features from the middle block
    
    with torch.no_grad():
        features = get_features(feature_extractor, dummy_input, dummy_time)
        feature_dim = features.shape[1] * features.shape[2] * features.shape[3]
    
    classifier = FeatureClassifier(feature_dim=feature_dim, num_classes=10).to(args.device)
    
    # --- 4. Train the Classifier ---
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    pbar = tqdm(range(args.train_steps))
    data_iter = cycle(train_dataloader)
    
    for i in pbar:
        classifier.train()
        optimizer.zero_grad()
        
        x, y = next(data_iter)
        x, y = x.to(args.device), y.to(args.device)
        
        # We use t=0 to extract features from the original, non-noised images
        t = torch.zeros(x.shape[0], dtype=torch.long, device=args.device)
        
        with torch.no_grad():
            features = get_features(feature_extractor, x, t)
            
        logits = classifier(features)
        loss = criterion(logits, y)
        
        loss.backward()
        optimizer.step()
        
        pbar.set_description(f"Classifier Loss: {loss.item():.4f}")

    print("Classifier training complete.")

    # --- 5. Evaluate the Classifier ---
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(test_dataloader, desc="Evaluating"):
            x, y = x.to(args.device), y.to(args.device)
            t = torch.zeros(x.shape[0], dtype=torch.long, device=args.device)
            
            features = get_features(feature_extractor, x, t)
            logits = classifier(features)
            
            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    print("="*40)
    print(f"Final Classification Accuracy: {accuracy:.2f}%")
    print("="*40)


if __name__ == "__main__":
    main()
