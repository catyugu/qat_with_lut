import torch
import torch.nn as nn
import numpy as np
import struct
import os

# Import model definition from the training script to avoid duplication
from train_qat_mlp import SmallMLP, TERNARY_THRESHOLD, TernaryQuantizeSTE

def convert_mlp_to_simple_binary():
    model = SmallMLP()
    model_path = 'small_mlp_ternary_act_qat.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please run train_qat_mlp.py first.")
        return

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    output_path = "mlp_model_aq.bin"
    print(f"Exporting model to: {output_path}")

    with open(output_path, 'wb') as f:
        f.write(b'MLP3')
        input_dim, hidden_dim, output_dim = 784, 320, 10
        f.write(struct.pack('iii', input_dim, hidden_dim, output_dim))
        state_dict = model.state_dict()

        # Layer 1 weights and bias
        w1_float = state_dict['fc1.weight']
        w1_ternary = TernaryQuantizeSTE.apply(w1_float.clamp(-1.0, 1.0)).numpy().astype(np.int8)
        f.write(w1_ternary.tobytes())
        f.write(state_dict['fc1.bias'].numpy().astype(np.float32).tobytes())

        # Activation Scales
        # Scale for input to first layer (images normalized to [-1, 1])
        input_to_fc1_scale = 127.0
        f.write(struct.pack('f', input_to_fc1_scale))

        # --- EXPORT LEARNED SCALE ---
        # Scale for input to second layer (activations from fc1)
        # This is now calculated based on the learned `running_abs_max` from our QAT module.
        learned_abs_max = state_dict['activation.running_abs_max'].item()
        input_to_fc2_scale = 127.0 / learned_abs_max
        print(f"Exporting learned `input_to_fc2_scale`: {input_to_fc2_scale:.4f}")
        f.write(struct.pack('f', input_to_fc2_scale))

        # Layer 2 weights and bias
        w2_float = state_dict['fc2.weight']
        w2_ternary = TernaryQuantizeSTE.apply(w2_float.clamp(-1.0, 1.0)).numpy().astype(np.int8)
        f.write(w2_ternary.tobytes())
        f.write(state_dict['fc2.bias'].numpy().astype(np.float32).tobytes())

    print(f"\nModel successfully exported to {output_path}.")

if __name__ == '__main__':
    convert_mlp_to_simple_binary()
