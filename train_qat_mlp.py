import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os

TERNARY_THRESHOLD = 0.001

# Ternary Quantize STE (用于权重和激活)
# 与 convert_qat.py 中的定义保持一致
class TernaryQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        return torch.where(input_tensor > TERNARY_THRESHOLD, 1.0,
                           torch.where(input_tensor < -TERNARY_THRESHOLD, -1.0, 0.0))
    @staticmethod
    def backward(ctx, grad_output):
        # 在反向传播时，量化器通常是直通的，梯度直接通过
        return grad_output

# Ternary Linear Layer (用于权重三值化)
# 与 convert_qat.py 中的定义保持一致
class TernaryLinear(nn.Linear):
    def forward(self, input):
        # 裁剪权重（与训练脚本保持一致）
        # 在应用三值化之前，通常会将权重裁剪到 -1 到 1 的范围，以匹配三值化的输出范围。
        clipped_weight = self.weight.clamp(-1.0, 1.0)
        # 应用三值化函数到裁剪后的权重
        quantized_weight = TernaryQuantizeSTE.apply(clipped_weight)
        return F.linear(input, quantized_weight, self.bias)

# Ternary Activation STE (用于激活三值化)
# 与 convert_qat.py 中的定义保持一致
class TernaryActivationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        return torch.where(input_tensor > TERNARY_THRESHOLD, 1.0,
                           torch.where(input_tensor < -TERNARY_THRESHOLD, -1.0, 0.0))
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# Quantized MLP Model
# 与 convert_qat.py 中的 SmallMLP 定义保持一致
class SmallMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=320, output_dim=10):
        super(SmallMLP, self).__init__()
        self.fc1 = TernaryLinear(input_dim, hidden_dim)
        # 直接使用 TernaryActivationSTE.apply 作为激活函数
        self.ternary_act_func = TernaryActivationSTE.apply
        self.fc2 = TernaryLinear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, 784) # 展平输入图像
        x = self.fc1(x)                # 第一个三值化线性层
        x = self.ternary_act_func(x)   # 三值化激活
        x = self.fc2(x)                # 第二个三值化线性层
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
    learning_rate = 0.0001
    criterion = nn.CrossEntropyLoss()

    # --- 训练量化感知模型 (Quantization-Aware Training) ---
    print("\n--- 训练量化感知模型 (Ternary Weight & Activation MLP) ---")
    qat_model = SmallMLP()
    qat_optimizer = torch.optim.Adam(qat_model.parameters(), lr=learning_rate)

    for e in range(epochs):
        qat_model.train()
        running_loss = 0; correct_train_preds = 0; total_train_preds = 0
        for images, labels in trainloader:
            qat_optimizer.zero_grad()
            output = qat_model(images)
            loss = criterion(output, labels)
            loss.backward()
            qat_optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train_preds += labels.size(0)
            correct_train_preds += (predicted == labels).sum().item()
        train_accuracy = 100 * correct_train_preds / total_train_preds

        qat_model.eval()
        test_correct_preds = 0; test_total_preds = 0
        with torch.no_grad():
            for images, labels in testloader:
                output = qat_model(images)
                _, predicted = torch.max(output.data, 1)
                test_total_preds += labels.size(0)
                test_correct_preds += (predicted == labels).sum().item()
        test_accuracy = 100 * test_correct_preds / test_total_preds
        print(f"Epoch {e+1}/{epochs} (QAT).. Train Loss: {running_loss/len(trainloader):.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")

    # 保存量化感知训练后的模型
    torch.save(qat_model.state_dict(), 'small_mlp_ternary_act_qat.pth')
    print("量化感知模型已保存到 'small_mlp_ternary_act_qat.pth'")

if __name__ == '__main__':
    train_models()