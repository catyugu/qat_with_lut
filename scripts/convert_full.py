import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import os

# 确保此模型定义与训练全精度模型时使用的定义完全一致
class FullPrecisionMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=320, output_dim=10):
        super(FullPrecisionMLP, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def convert_float_model_to_binary():
    model = FullPrecisionMLP()
    # 确保此路径指向您训练的全精度模型文件
    model_path = 'small_mlp_float.pth'
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件 '{model_path}'。请先运行 train_my_mlp.py 来训练全精度模型。")
        return

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval() # 设置为评估模式

    output_path = "mlp_model_float.bin"
    print(f"正在导出全精度模型到: {output_path}")

    with open(output_path, 'wb') as f:
        # 1. 写入头部 (Magic number: 'MLPF' for MLP Float)
        f.write(b'MLPF') # 版本或模型类型标识符

        # 2. 写入维度
        input_dim = 784
        hidden_dim = 320
        output_dim = 10
        f.write(struct.pack('iii', input_dim, hidden_dim, output_dim))

        state_dict = model.state_dict()

        # --- 第一层权重和偏置 (全精度 float32) ---
        w1_float = state_dict['fc1.weight'].numpy().astype(np.float32)
        b1_float = state_dict['fc1.bias'].numpy().astype(np.float32)
        f.write(w1_float.tobytes())
        f.write(b1_float.tobytes())

        # --- 激活尺度 (对于全精度模型，均为 1.0f) ---
        # input_to_fc1_scale 和 input_to_fc2_scale 实际上用于量化模型
        # 对于全精度模型，它们仅作为占位符，值为 1.0f
        input_to_fc1_scale = 1.0 # 理论上没有量化，但为了文件结构一致性而保留
        input_to_fc2_scale = 1.0 # 理论上没有量化，但为了文件结构一致性而保留
        f.write(struct.pack('f', input_to_fc1_scale))
        f.write(struct.pack('f', input_to_fc2_scale))


        # --- 第二层权重和偏置 (全精度 float32) ---
        w2_float = state_dict['fc2.weight'].numpy().astype(np.float32)
        b2_float = state_dict['fc2.bias'].numpy().astype(np.float32)
        f.write(w2_float.tobytes())
        f.write(b2_float.tobytes())

    print("\n全精度模型成功导出到 mlp_model_float.bin。")

if __name__ == '__main__':
    convert_float_model_to_binary()
