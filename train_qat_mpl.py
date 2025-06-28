import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os

# 定义用于权重和激活的三值量化阈值
# 这是关键参数，可能需要根据训练结果进行微调。
# 初始设置为0.5，旨在避免过多值为0，让梯度更有效传播。
TERNARY_THRESHOLD = 0.01

class TernaryQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        # 量化输入张量为三值（-1, 0, 1）
        # 大于阈值的变为1.0，小于负阈值的变为-1.0，否则为0.0
        return torch.where(input_tensor > TERNARY_THRESHOLD, 1.0,
                           torch.where(input_tensor < -TERNARY_THRESHOLD, -1.0, 0.0))

    @staticmethod
    def backward(ctx, grad_output):
        # 直通估计器（STE）：梯度直接通过非可微的量化操作
        return grad_output, None # None for the input_tensor argument


class TernaryLinear(nn.Linear):
    def forward(self, input):
        # 在应用三值量化之前，先对权重进行裁剪
        # 这有助于稳定训练，避免权重幅度过大
        clipped_weight = self.weight.clamp(-1.0, 1.0)

        # 将裁剪后的权重应用三值量化
        # 存储的self.weight仍然是浮点数，用于梯度更新
        quantized_weight = TernaryQuantizeSTE.apply(clipped_weight)

        return F.linear(input, quantized_weight, self.bias)

# Ternary Activation STE 用于直接将激活量化为 -1, 0, 1
class TernaryActivationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        # 量化激活为三值（-1, 0, 1）
        return torch.where(input_tensor > TERNARY_THRESHOLD, 1.0,
                           torch.where(input_tensor < -TERNARY_THRESHOLD, -1.0, 0.0))

    @staticmethod
    def backward(ctx, grad_output):
        # 直通估计器
        return grad_output, None

class SmallMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=320, output_dim=10):
        super(SmallMLP, self).__init__()
        self.input_dim = input_dim
        self.fc1 = TernaryLinear(input_dim, hidden_dim)
        self.ternary_act_func = TernaryActivationSTE.apply
        self.fc2 = TernaryLinear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim) # 展平输入图像
        x = self.fc1(x)           # 第一个带有三值权重的线性层
        # 移除F.relu(x)。三值量化本身提供了非线性。
        # 应用三值量化到激活上
        x = self.ternary_act_func(x) # 不再传递阈值，直接使用全局定义的TERNARY_THRESHOLD
        x = self.fc2(x)           # 第二个带有三值权重的线性层
        return F.log_softmax(x, dim=1) # LogSoftmax 用于分类输出

def train_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 数据路径设置：假设FashionMNIST数据将下载到'datasets/F_MNIST_data'文件夹
    data_path = os.path.join(script_dir, 'datasets', 'F_MNIST_data')
    os.makedirs(data_path, exist_ok=True) # 如果目录不存在则创建

    # 定义FashionMNIST图像的转换
    # ToTensor()将PIL图像转换为FloatTensor并缩放到[0.0, 1.0]
    # Normalize()使用均值0.5和标准差0.5缩放到[-1.0, 1.0]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # 加载训练数据集
    trainset = datasets.FashionMNIST(data_path, download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # 加载测试数据集用于训练期间的评估
    testset = datasets.FashionMNIST(data_path, download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    model = SmallMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss() # 适用于LogSoftmax输出与标签结合

    epochs = 3 # 增加训练周期以观察效果
    print(f"开始使用权重和三值激活量化（QAT）进行训练，阈值为：{TERNARY_THRESHOLD}...")
    for e in range(epochs):
        model.train() # 设置模型为训练模式
        running_loss = 0; correct_train_preds = 0; total_train_preds = 0
        for images, labels in trainloader:
            optimizer.zero_grad() # 清除梯度
            output = model(images) # 前向传播
            loss = criterion(output, labels) # 计算损失
            loss.backward() # 反向传播（计算梯度）
            optimizer.step() # 更新权重

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1) # 获取预测类别
            total_train_preds += labels.size(0)
            correct_train_preds += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train_preds / total_train_preds
        print(f"Epoch {e+1}/{epochs}.. Train Loss: {running_loss/len(trainloader):.4f}, Train Acc: {train_accuracy:.2f}%")

        # 每个epoch后在测试集上评估模型
        model.eval() # 设置模型为评估模式（禁用dropout，批量归一化更新等）
        test_correct_preds = 0; test_total_preds = 0
        with torch.no_grad(): # 评估时禁用梯度计算
            for images, labels in testloader:
                output = model(images)
                _, predicted = torch.max(output.data, 1)
                test_total_preds += labels.size(0)
                test_correct_preds += (predicted == labels).sum().item()

        test_accuracy = 100 * test_correct_preds / test_total_preds
        # 此test_accuracy现在反映了在推理过程中带有三值权重和三值激活的模型的性能。
        print(f"Epoch {e+1}/{epochs}.. Test Acc (QAT Ternary Act): {test_accuracy:.2f}%")

    print("\n训练完成。")
    # 保存state_dict（权重和激活量化器尺度）
    # 保存的state_dict将包含浮点权重，需要为C++推理进行三值量化。
    torch.save(model.state_dict(), 'small_mlp_ternary_act_qat.pth')
    print("模型已保存到'small_mlp_ternary_act_qat.pth'")

if __name__ == '__main__':
    train_model()
