import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from ddpm.qat_classifier import QAT_Classifier 

def main():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./cifar_train', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    testset = torchvision.datasets.CIFAR10(root='./cifar_test', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # 3. Model, Loss, and Optimizer
    model = QAT_Classifier(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Separate parameters for different learning rates
    scale_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'scale' in name:
            scale_params.append(param)
        else:
            other_params.append(param)

    # Set up the optimizer with two parameter groups
    optimizer = optim.AdamW([
        {'params': other_params},
        {'params': scale_params, 'lr': 5e-3}  # Higher learning rate for scale parameters
    ], lr=5e-4, weight_decay=1e-6) # Base learning rate for other parameters
    
    # --- Training Configuration ---
    num_epochs = 35 
    warmup_epochs = 10  # Number of epochs to train before enabling quantization
    
    print(f"Starting training for {num_epochs} epochs with a {warmup_epochs} epoch warm-up...")

    # 4. Training Loop
    for epoch in range(1, num_epochs + 1):
        # --- Warm-up and Learning Rate Schedule Logic ---
        is_quant_epoch = epoch > warmup_epochs
        
        # At the end of the warm-up, reduce the learning rate for stable fine-tuning
        if epoch == warmup_epochs + 1:
            print("\n--- Warm-up finished. Quantization is now ENABLED. ---")
        # Enable quantization on all relevant modules
        for module in model.modules():
            if hasattr(module, 'quantize'):
                module.quantize = is_quant_epoch

        # --- Train for one epoch ---
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(trainloader)

        # --- Evaluate on the test set ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        
        # --- Print Epoch Statistics ---
        try:
            scale1 = model.q_act1.scale.item()
            scale4 = model.q_act4.scale.item()
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}% | Scales: q1={scale1:.3f}, q4={scale4:.3f}')
        except AttributeError:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')


    print('Finished Training')
    torch.save(model.state_dict(), 'qat_classifier_cifar10_final.pth')
    print('Model saved to qat_classifier_cifar10_final.pth')


if __name__ == '__main__':
    main()
