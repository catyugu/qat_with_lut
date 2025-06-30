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
        # Use a new magic number 'MLP4' for the updated format
        f.write(b'MLP4')
        input_dim, hidden_dim, output_dim = 784, 320, 10
        f.write(struct.pack('iii', input_dim, hidden_dim, output_dim))
        state_dict = model.state_dict()

        # Layer 1 weights and bias
        w1_float = state_dict['fc1.weight']
        w1_ternary = TernaryQuantizeSTE.apply(w1_float.clamp(-1.0, 1.0)).numpy().astype(np.int8)
        f.write(w1_ternary.tobytes())
        f.write(state_dict['fc1.bias'].numpy().astype(np.float32).tobytes())

        # --- EXPORT THE SINGLE LEARNED ACTIVATION SCALE ---
        # This is the scale for the activations between fc1 and fc2.
        # The scale for the input layer is fixed and will be hardcoded in the C++ app.
        learned_abs_max = state_dict['activation.running_abs_max'].item()
        hidden_activation_scale = 127.0 / learned_abs_max
        print(f"Exporting the single learned hidden activation scale: {hidden_activation_scale:.4f}")
        f.write(struct.pack('f', hidden_activation_scale))

        # Layer 2 weights and bias
        w2_float = state_dict['fc2.weight']
        w2_ternary = TernaryQuantizeSTE.apply(w2_float.clamp(-1.0, 1.0)).numpy().astype(np.int8)
        f.write(w2_ternary.tobytes())
        f.write(state_dict['fc2.bias'].numpy().astype(np.float32).tobytes())

    print(f"\nModel successfully exported to {output_path} with the new format (MLP4).")

if __name__ == '__main__':
    convert_mlp_to_simple_binary()
