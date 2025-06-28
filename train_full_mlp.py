import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
class FullPrecisionMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=320, output_dim=10):
        super(FullPrecisionMLP, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim) # 展平输入图像
        x = self.fc1(x)                # 第一个全精度线性层
        x = F.relu(x)                  # ReLU 激活
        x = self.fc2(x)                # 第二个全精度线性层
        return F.log_softmax(x, dim=1) # LogSoftmax 用于分类输出


def train_models():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'datasets', 'F_MNIST_data')
    os.makedirs(data_path, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    trainset = datasets.FashionMNIST(data_path, download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = datasets.FashionMNIST(data_path, download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    epochs = 5
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()

    # --- 训练全精度模型 (Full Precision) ---
    print("\n--- 训练全精度模型 (Standard Float MLP) ---")
    float_model = FullPrecisionMLP()
    float_optimizer = torch.optim.Adam(float_model.parameters(), lr=learning_rate)

    for e in range(epochs):
        float_model.train()
        running_loss = 0; correct_train_preds = 0; total_train_preds = 0
        for images, labels in trainloader:
            float_optimizer.zero_grad()
            output = float_model(images)
            loss = criterion(output, labels)
            loss.backward()
            float_optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train_preds += labels.size(0)
            correct_train_preds += (predicted == labels).sum().item()
        train_accuracy = 100 * correct_train_preds / total_train_preds

        float_model.eval()
        test_correct_preds = 0; test_total_preds = 0
        with torch.no_grad():
            for images, labels in testloader:
                output = float_model(images)
                _, predicted = torch.max(output.data, 1)
                test_total_preds += labels.size(0)
                test_correct_preds += (predicted == labels).sum().item()
        test_accuracy = 100 * test_correct_preds / test_total_preds
        print(f"Epoch {e+1}/{epochs} (Float).. Train Loss: {running_loss/len(trainloader):.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")

    torch.save(float_model.state_dict(), 'small_mlp_float.pth')
    print("全精度模型已保存到 'small_mlp_float.pth'")

if __name__ == '__main__':
    train_models()
