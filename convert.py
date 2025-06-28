import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import os

# Ensure these classes match your train_my_mlp.py
class TernaryQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, threshold=0.1):
        return torch.where(input_tensor > threshold, 1.0, torch.where(input_tensor < -0.1, -1.0, 0.0))
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class TernaryLinear(nn.Linear):
    def forward(self, input):
        quantized_weight = TernaryQuantizeSTE.apply(self.weight)
        return F.linear(input, quantized_weight, self.bias)

class ActivationQuantizer(nn.Module):
    def __init__(self):
        super(ActivationQuantizer, self).__init__()
        self.register_buffer('scale', torch.tensor(1.0, dtype=torch.float32)) # This scale is dynamically updated
    def forward(self, x):
        if self.training:
            self.scale.data[()] = 127.0 / x.abs().max()
        x_quant = (x * self.scale).round().clamp(-128, 127) # Clamp to int8 range
        x_dequant = x_quant / self.scale
        return x + (x_dequant - x).detach()

class SmallMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super(SmallMLP, self).__init__()
        self.fc1 = TernaryLinear(input_dim, hidden_dim)
        self.act_quant1 = ActivationQuantizer() # This quantizes output of fc1+relu
        self.fc2 = TernaryLinear(hidden_dim, output_dim)
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        # Activation quantization after ReLU, before fc2
        x = self.act_quant1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def convert_mlp_to_simple_binary():
    model = SmallMLP()
    # Ensure this path is correct for your trained model
    model_path = 'small_mlp_full_qat.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please run train_my_mlp.py first.")
        return

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval() # Set to evaluation mode

    output_path = "mlp_model_aq.bin"
    print(f"Exporting model to: {output_path}")

    with open(output_path, 'wb') as f:
        # 1. Write Header
        f.write(b'MLP3') # Version 3

        # 2. Write dimensions
        input_dim = 784
        hidden_dim = 256
        output_dim = 10
        f.write(struct.pack('iii', input_dim, hidden_dim, output_dim))

        state_dict = model.state_dict()

        # --- Layer 1 Weights and Bias ---
        w1_float = state_dict['fc1.weight']
        # Convert ternary floats (1.0, -1.0, 0.0) to int8 (1, -1, 0)
        w1_ternary = torch.where(w1_float > 0.1, 1.0, torch.where(w1_float < -0.1, -1.0, 0.0)).numpy().astype(np.int8)
        f.write(w1_ternary.tobytes())
        f.write(state_dict['fc1.bias'].numpy().astype(np.float32).tobytes())

        # --- Activation Scales ---
        # Scale for input to first layer (images normalized to [-1, 1])
        input_to_fc1_scale = 127.0 # Max abs value of normalized FashionMNIST is 1.0
        f.write(struct.pack('f', input_to_fc1_scale))

        # Scale for input to second layer (output of fc1 + relu, then ActivationQuantizer)
        # This scale is stored in the ActivationQuantizer's buffer during QAT.
        input_to_fc2_scale = state_dict['act_quant1.scale'].item() # Get the scalar float value
        f.write(struct.pack('f', input_to_fc2_scale))


        # --- Layer 2 Weights and Bias ---
        w2_float = state_dict['fc2.weight']
        # Convert ternary floats (1.0, -1.0, 0.0) to int8 (1, -1, 0)
        w2_ternary = torch.where(w2_float > 0.1, 1.0, torch.where(w2_float < -0.1, -1.0, 0.0)).numpy().astype(np.int8)
        f.write(w2_ternary.tobytes())
        f.write(state_dict['fc2.bias'].numpy().astype(np.float32).tobytes())

    print("\nModel successfully exported to mlp_model_aq.bin.")

if __name__ == '__main__':
    convert_mlp_to_simple_binary()
