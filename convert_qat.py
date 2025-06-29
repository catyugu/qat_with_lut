import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import os

# 确保此阈值与train_my_mlp.py中定义的TERNARY_THRESHOLD一致
TERNARY_THRESHOLD = 0.001

class TernaryQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor): # 移除了threshold参数，使用全局常量
        return torch.where(input_tensor > TERNARY_THRESHOLD, 1.0,
                           torch.where(input_tensor < -TERNARY_THRESHOLD, -1.0, 0.0))
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class TernaryLinear(nn.Linear):
    def forward(self, input):
        # 裁剪权重（与训练脚本保持一致）
        clipped_weight = self.weight.clamp(-1.0, 1.0)
        quantized_weight = TernaryQuantizeSTE.apply(clipped_weight)
        return F.linear(input, quantized_weight, self.bias)

# Ternary Activation STE (与训练脚本保持一致)
class TernaryActivationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor): # 移除了threshold参数，使用全局常量
        return torch.where(input_tensor > TERNARY_THRESHOLD, 1.0,
                           torch.where(input_tensor < -TERNARY_THRESHOLD, -1.0, 0.0))
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class SmallMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=320, output_dim=10):
        super(SmallMLP, self).__init__()
        self.fc1 = TernaryLinear(input_dim, hidden_dim)
        self.ternary_act_func = TernaryActivationSTE.apply
        self.fc2 = TernaryLinear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.ternary_act_func(x) # 不再传递阈值
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def convert_mlp_to_simple_binary():
    model = SmallMLP()
    # 确保此路径与您训练的模型路径一致
    model_path = 'small_mlp_ternary_act_qat.pth'
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件 '{model_path}'。请先运行 train_my_mlp.py。")
        return

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval() # 设置为评估模式

    output_path = "mlp_model_aq.bin"
    print(f"正在导出模型到: {output_path}")

    with open(output_path, 'wb') as f:
        # 1. 写入头部
        f.write(b'MLP3') # 版本 3

        # 2. 写入维度
        input_dim = 784
        hidden_dim = 320
        output_dim = 10
        f.write(struct.pack('iii', input_dim, hidden_dim, output_dim))

        state_dict = model.state_dict()

        # --- 第一层权重和偏置 ---
        w1_float = state_dict['fc1.weight']
        # 在转换为int8之前，应用与训练时相同的裁剪和三值化逻辑
        w1_ternary = torch.where(w1_float.clamp(-1.0, 1.0) > TERNARY_THRESHOLD, 1.0,
                                 torch.where(w1_float.clamp(-1.0, 1.0) < -TERNARY_THRESHOLD, -1.0, 0.0)).numpy().astype(np.int8)
        f.write(w1_ternary.tobytes())
        f.write(state_dict['fc1.bias'].numpy().astype(np.float32).tobytes())

        # --- 激活尺度 ---
        # 第一层输入的尺度（图像已归一化到[-1, 1]）
        # 此尺度在C++中用于将浮点输入初步量化为int8，然后转换为三值。
        input_to_fc1_scale = 127.0 # FashionMNIST归一化后的最大绝对值为1.0
        f.write(struct.pack('f', input_to_fc1_scale))

        # 第二层输入的尺度（fc1 + TernaryActivationSTE 的输出）
        # 由于激活现在直接在Python中三值化为{-1.0, 0.0, 1.0}，
        # 因此在C++中用于反量化累加和的尺度将是1.0。
        input_to_fc2_scale = 1.0 # 激活现在是{-1, 0, 1}
        f.write(struct.pack('f', input_to_fc2_scale))


        # --- 第二层权重和偏置 ---
        w2_float = state_dict['fc2.weight']
        # 在转换为int8之前，应用与训练时相同的裁剪和三值化逻辑
        w2_ternary = torch.where(w2_float.clamp(-1.0, 1.0) > TERNARY_THRESHOLD, 1.0,
                                 torch.where(w2_float.clamp(-1.0, 1.0) < -TERNARY_THRESHOLD, -1.0, 0.0)).numpy().astype(np.int8)
        f.write(w2_ternary.tobytes())
        f.write(state_dict['fc2.bias'].numpy().astype(np.float32).tobytes())

    print("\n模型成功导出到 mlp_model_aq.bin。")

if __name__ == '__main__':
    convert_mlp_to_simple_binary()
