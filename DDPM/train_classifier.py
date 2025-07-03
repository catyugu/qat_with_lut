# In DDPM/train_classifier.py

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
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./cifar_test', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # 3. Model, Loss, and Optimizer
    model = QAT_Classifier(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # --- STABILITY FIX: Lower learning rate ---
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Lowered from 0.001
    
    num_epochs = 50 # Increased epochs to see learning progression
    print("Starting training with corrected gradients and lower learning rate...")

    # 4. Training Loop
    for epoch in range(num_epochs):
        model.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 5. Evaluation
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        
        # --- DEBUGGING: Print the learned scale parameters ---
        scale1 = model.q_act1.scale.item()
        scale4 = model.q_act4.scale.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}% | Scales: q1={scale1:.3f}, q4={scale4:.3f}')


    print('Finished Training')
    torch.save(model.state_dict(), 'qat_classifier_cifar10_fixed.pth')
    print('Model saved to qat_classifier_cifar10_fixed.pth')


if __name__ == '__main__':
    main()