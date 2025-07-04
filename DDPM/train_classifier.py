import os
import torch
import argparse
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

# Import the necessary components from your project structure
# QATLinear and QAT_Classifier are now imported directly from ddpm.qat_classifier
from ddpm.qat_classifier import QAT_Classifier, QATLinear # Import QATLinear if needed elsewhere, or remove if only used by QAT_Classifier

def main():
    parser = argparse.ArgumentParser(description="Train a QAT Classifier on CIFAR-10.")
    
    # Paths
    parser.add_argument("--dataset_path", type=str, default="./data", help="Path to CIFAR-10 dataset")
    parser.add_argument("--save_model_path", type=str, default="./qat_classifier.pth", help="Path to save the trained classifier model.")
    
    # Training Hyperparameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training (e.g., 'cuda', 'cpu')")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--quantize", action="store_true", help="Enable quantization during training.")
    
    args = parser.parse_args()
    
    print("="*40)
    print("Starting QAT Classifier Training")
    print(f"Quantization Enabled: {args.quantize}")
    print(f"Device: {args.device}")
    print("="*40)

    # --- 1. Load Datasets ---
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=args.dataset_path, train=True, download=True, transform=transform
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.dataset_path, train=False, download=True, transform=transform
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- 2. Initialize the Classifier ---
    model = QAT_Classifier(num_classes=10).to(args.device) # Use the imported QAT_Classifier
    
    # Set quantization state based on argument
    model.set_quantization_enabled(args.quantize)

    # --- 3. Train the Classifier ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = 0.0

    for epoch in range(args.epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss/(i+1))

        # --- 4. Evaluate the Classifier ---
        model.eval() # Set model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Eval]"):
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1} - Training Loss: {running_loss / len(train_dataloader):.4f}, Test Accuracy: {accuracy:.2f}%")

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), args.save_model_path)
            print(f"Saved best model with accuracy: {best_accuracy:.2f}% to {args.save_model_path}")

    print("="*40)
    print("Classifier training complete.")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print("="*40)


if __name__ == "__main__":
    main()

