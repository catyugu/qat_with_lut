import torch
import torch.nn as nn
import numpy as np
import struct
import os
import sys
import argparse # Import argparse to create a dummy args object

# Add the project root to sys.path to allow imports from DDPM
# This assumes the script is run from the 'scripts' directory
# and 'DDPM' is in the parent directory of 'scripts'.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Ensure these imports match your project structure
from DDPM.ddpm.QAT_UNet import QATUNet 
from DDPM.ddpm.script_utils_qat import diffusion_defaults # Only need diffusion_defaults for default options
from DDPM.ddpm.QAT_UNet import ScaledWeightTernary, ScaledActivationTernary

# --- Helper functions for quantization and packing (from your existing utils.py logic) ---

def ternarize_tensor_to_int8(x, alpha):
    """Ternarizes a float tensor to -1, 0, 1 based on alpha threshold."""
    # Assuming alpha is the mean absolute value, and threshold is 0.05 * alpha
    # Since we set C_TERNARY_ACTIVATION_THRESHOLD = 0, we'll use a hard 0 threshold.
    # For weights, the ScaledWeightTernary handles its own alpha.
    # Here, we're just making sure the output is -1, 0, or 1.
    x_ternary = torch.where(x > 0, torch.ones_like(x), torch.where(x < 0, -torch.ones_like(x), torch.zeros_like(x)))
    return x_ternary.to(torch.int8)

def pack_ternary_weights_5x3bit(tensor):
    """Packs a 1D tensor of -1, 0, 1 (int8) into a 5x3bit format (uint8).
    Each uint8 byte stores 5 ternary values.
    The mapping is: -1 -> 0, 0 -> 1, 1 -> 2
    """
    # Ensure tensor is 1D and contains only -1, 0, 1
    assert tensor.dim() == 1
    assert torch.all((tensor >= -1) & (tensor <= 1))

    # Map -1, 0, 1 to 0, 1, 2
    mapped_tensor = (tensor + 1).to(torch.uint8) # -1->0, 0->1, 1->2

    num_elements = mapped_tensor.numel()
    # Calculate bytes needed: ceil(num_elements * 3 / 8)
    num_bytes = (num_elements * 3 + 7) // 8
    packed_data = bytearray(num_bytes)

    for i in range(num_elements):
        val = mapped_tensor[i].item()
        bit_offset = i * 3
        byte_idx = bit_offset // 8
        bit_in_byte_offset = bit_offset % 8

        # Pack 3 bits into the current byte
        packed_data[byte_idx] |= (val << bit_in_byte_offset)
        if bit_in_byte_offset + 3 > 8: # If spills into next byte
            packed_data[byte_idx + 1] |= (val >> (8 - bit_in_byte_offset))

    return bytes(packed_data)

# --- Main Export Logic ---

def export_qat_unet_to_binary(model_path, output_dir="exported_unet_qat_model"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "qat_unet_model.bin")

    # Load the model configuration (adjust as per your training script)
    model_options = diffusion_defaults()
    model_options['image_size'] = 32 # CIFAR-10
    model_options['in_channels'] = 3 # Assuming RGB images for CIFAR-10
    model_options['num_channels'] = 128 # Base channels
    model_options['num_res_blocks'] = 2
    model_options['num_heads'] = 4 # Default in QAT_UNet
    model_options['num_heads_upsample'] = -1 # Default
    model_options['num_head_channels'] = 64 # Default
    model_options['attention_resolutions'] = "16,8" # Default
    model_options['channel_mult'] = "1,2,2,2" # Default
    model_options['dropout'] = 0.1
    model_options['class_cond'] = True # CIFAR-10 is class conditional
    model_options['use_scale_shift_norm'] = True # Default
    model_options['resblock_updown'] = False # Default
    model_options['use_fp16'] = False # Not using fp16 for QAT
    model_options['initial_pad'] = 0 # Default
    model_options['use_new_attention_architecture'] = False # Default

    # Add missing arguments for QATUNet constructor
    model_options['time_emb_scale'] = 1.0 # Default scale, adjust if your training uses a different one
    model_options['num_groups'] = 32 # Common default for GroupNorm, adjust if your QAT_UNet uses a different value

    # Parse attention resolutions and channel multipliers
    model_options['attention_resolutions'] = [int(x) for x in model_options['attention_resolutions'].split(',')]
    model_options['channel_mult'] = [int(x) for x in model_options['channel_mult'].split(',')]
    
    # Determine num_classes
    model_options['num_classes'] = 10 if model_options['class_cond'] else None

    # Create the model instance directly, passing all required arguments
    model = QATUNet(
        img_channels=model_options['in_channels'],
        base_channels=model_options['num_channels'],
        channel_mults=model_options['channel_mult'],
        num_res_blocks=model_options['num_res_blocks'],
        time_emb_dim=model_options['time_emb_dim'],
        time_emb_scale=model_options['time_emb_scale'],
        num_classes= model_options['num_classes'] ,
        dropout=model_options['dropout'],
        attention_resolutions=model_options['attention_resolutions'],
        norm=model_options['norm'], # 'gn' for GroupNorm
        num_groups=model_options['num_groups'],
        initial_pad=model_options['initial_pad']
    )

    # Load the trained model state dict with strict=False
    print(f"Loading model from {model_path} with strict=False...")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False) # <--- Added strict=False
    model.eval() # Set to evaluation mode

    print("Model loaded successfully. Starting export...")

    with open(output_file_path, 'wb') as f:
        # --- Write Model Hyperparameters (for C++ to reconstruct architecture) ---
        # Changed to '<I' for unsigned int
        f.write(struct.pack('<I', model_options['image_size']))
        f.write(struct.pack('<I', model_options['in_channels'])) # Usually 3 for RGB
        f.write(struct.pack('<I', model_options['num_channels'])) # base_channels
        f.write(struct.pack('<I', len(model_options['channel_mult'])))
        for mult in model_options['channel_mult']:
            f.write(struct.pack('<I', mult))
        f.write(struct.pack('<I', model_options['num_res_blocks']))
        f.write(struct.pack('<I', model_options['time_emb_dim']))
        f.write(struct.pack('<f', model_options['time_emb_scale']))
        f.write(struct.pack('<I', model_options['num_classes'] if model_options['num_classes'] is not None else 0)) # Write 0 if no classes
        f.write(struct.pack('<f', model_options['dropout']))
        f.write(struct.pack('<I', len(model_options['attention_resolutions'])))
        for res in model_options['attention_resolutions']:
            f.write(struct.pack('<I', res))
        f.write(struct.pack('<I', model_options['num_groups'])) # For GroupNorm
        f.write(struct.pack('<I', model_options['initial_pad']))
        f.write(struct.pack('<I', 1 if model_options['use_scale_shift_norm'] else 0)) # Boolean as int

        # --- Iterate through model layers and export parameters ---
        # This will be complex due to the nested structure of UNet.
        # We need to traverse the model and identify QATConv2d and GroupNorm layers.

        # A helper to write a layer's parameters
        def write_conv_layer(layer_name, layer, file_handle):
            print(f"  Exporting layer: {layer_name}")
            assert isinstance(layer.weight_quant, ScaledWeightTernary), f"Expected ScaledWeightTernary for {layer_name}"

            # Get quantized weights and their scale
            weight_scale = layer.weight_quant._alpha.item()
            ternary_weights = ternarize_tensor_to_int8(layer.weight_quant.weight, weight_scale)
            packed_weights = pack_ternary_weights_5x3bit(ternary_weights.flatten())

            bias = layer.bias.detach().cpu().numpy() if layer.bias is not None else np.array([])

            # Debugging prints for sizes *before* packing into struct
            print(f"    DEBUG (Python): {layer_name} - raw packed_weights_len = {len(packed_weights)}")
            print(f"    DEBUG (Python): {layer_name} - raw bias_size = {bias.size}")

            # Write layer metadata (Changed to '<I')
            file_handle.write(struct.pack('<I', layer.in_channels))
            file_handle.write(struct.pack('<I', layer.out_channels))
            file_handle.write(struct.pack('<I', layer.kernel_size[0])) # kernel_h
            file_handle.write(struct.pack('<I', layer.kernel_size[1])) # kernel_w
            file_handle.write(struct.pack('<I', layer.stride[0]))     # stride_h
            file_handle.write(struct.pack('<I', layer.stride[1]))     # stride_w
            file_handle.write(struct.pack('<I', layer.padding[0]))    # pad_h
            file_handle.write(struct.pack('<I', layer.padding[1]))    # pad_w
            file_handle.write(struct.pack('<I', layer.groups))
            file_handle.write(struct.pack('<f', weight_scale))

            # Write packed weights (Changed to '<I')
            file_handle.write(struct.pack('<I', len(packed_weights)))
            file_handle.write(packed_weights)

            # Write bias (Changed to '<I')
            file_handle.write(struct.pack('<I', bias.size))
            file_handle.write(bias.tobytes())

        def write_groupnorm_layer(layer_name, layer, file_handle):
            print(f"  Exporting GroupNorm: {layer_name}")
            num_groups = layer.num_groups
            num_channels = layer.num_channels
            eps = layer.eps
            gamma = layer.weight.detach().cpu().numpy() if layer.weight is not None else np.array([])
            beta = layer.bias.detach().cpu().numpy() if layer.bias is not None else np.array([])

            # Debugging prints for sizes *before* packing into struct
            print(f"    DEBUG (Python): {layer_name} - raw num_groups = {num_groups}")
            print(f"    DEBUG (Python): {layer_name} - raw num_channels = {num_channels}")
            print(f"    DEBUG (Python): {layer_name} - raw gamma_size = {gamma.size}")
            print(f"    DEBUG (Python): {layer_name} - raw beta_size = {beta.size}")

            # Changed to '<I'
            file_handle.write(struct.pack('<I', num_groups))
            file_handle.write(struct.pack('<I', num_channels))
            file_handle.write(struct.pack('<f', eps))

            file_handle.write(struct.pack('<I', gamma.size))
            file_handle.write(gamma.tobytes())
            file_handle.write(struct.pack('<I', beta.size))
            file_handle.write(beta.tobytes())


        # Corrected: Pass file_handle to write_linear_layer
        def write_linear_layer(layer_name, layer, file_handle):
            print(f"  Exporting Linear layer: {layer_name}")
            weight = layer.weight.detach().cpu().numpy()
            bias = layer.bias.detach().cpu().numpy() if layer.bias is not None else np.array([])

            # Debugging prints for sizes *before* packing into struct
            print(f"    DEBUG (Python): {layer_name} - raw out_features = {weight.shape[0]}")
            print(f"    DEBUG (Python): {layer_name} - raw in_features = {weight.shape[1]}")
            print(f"    DEBUG (Python): {layer_name} - raw weight_size = {weight.size}")
            print(f"    DEBUG (Python): {layer_name} - raw bias_size = {bias.size}")

            # Changed to '<I'
            file_handle.write(struct.pack('<I', weight.shape[0])) # out_features
            file_handle.write(struct.pack('<I', weight.shape[1])) # in_features
            file_handle.write(struct.pack('<I', weight.size))
            file_handle.write(weight.tobytes())
            file_handle.write(struct.pack('<I', bias.size))
            file_handle.write(bias.tobytes())


        # Traverse the model recursively and export layers
        # Corrected: Accept file_handle as an argument
        def export_module(name, module, file_handle):
            if isinstance(module, nn.Conv2d) and hasattr(module, 'weight_quant'): # QATConv2d
                write_conv_layer(name, module, file_handle) # Corrected: Pass file_handle
            elif isinstance(module, nn.GroupNorm):
                write_groupnorm_layer(name, module, file_handle) # Corrected: Pass file_handle
            # Add other layer types if they have parameters to export (e.g., Linear for time embeddings)
            elif isinstance(module, nn.Linear):
                write_linear_layer(name, module, file_handle) # Corrected: Pass file_handle
            else:
                for child_name, child_module in module.named_children():
                    export_module(f"{name}.{child_name}", child_module, file_handle) # Corrected: Pass file_handle

        # Start exporting from the top-level model
        export_module("model", model, f) # Corrected: Pass f (file_handle)

    print(f"Model exported to {output_file_path}")
    print("Please note: This script is a starting point. You'll need to meticulously match")
    print("the C++ loading logic with the exact order and format of parameters saved here.")
    print("Complexities like skip connections, time embeddings, and class conditioning")
    print("will require careful C++ implementation mirroring the PyTorch model's forward pass.")


if __name__ == "__main__":
    # Replace with the actual path to your trained QAT UNet model checkpoint
    TRAINED_MODEL_PATH = "qat_unet_progressive.pth"
    # Example: TRAINED_MODEL_PATH = "logs/cifar10_qat_unet/model000000.pth"

    if not os.path.exists(TRAINED_MODEL_PATH):
        print(f"Error: Trained model not found at {TRAINED_MODEL_PATH}")
        print("Please update TRAINED_MODEL_PATH to your actual checkpoint file.")
    else:
        export_qat_unet_to_binary(TRAINED_MODEL_PATH)

