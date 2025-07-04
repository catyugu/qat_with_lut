import torch
import torch.nn as nn
import numpy as np
import struct
import os
import sys
import argparse

# Add the project root to sys.path to allow imports from DDPM
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Ensure these imports match your project structure
# Import the QATUNet model with the updated AttentionBlock
from DDPM.ddpm.QAT_UNet import QATUNet, ScaledWeightTernary, ScaledActivationTernary, QATConv2d, AttentionBlock, HadamardTransform, PositionalEmbedding

# Import diffusion_defaults for model configuration
from DDPM.ddpm.script_utils_qat import diffusion_defaults

# --- Helper functions for quantization and packing ---

def ternarize_tensor_to_int8(x):
    """Ternarizes a float tensor to -1, 0, 1."""
    x_ternary = torch.where(x > 0, torch.ones_like(x), torch.where(x < 0, -torch.ones_like(x), torch.zeros_like(x)))
    return x_ternary.to(torch.int8)

def pack_ternary_weights_5x3bit(tensor):
    """Packs a 1D tensor of -1, 0, 1 (int8) into a 5x3bit format (uint8).
    Each uint8 byte stores 5 ternary values.
    The mapping is: -1 -> 0, 0 -> 1, 1 -> 2
    """
    assert tensor.dim() == 1, f"Expected 1D tensor, got {tensor.dim()}D"
    assert torch.all((tensor >= -1) & (tensor <= 1)), f"Tensor contains values outside -1, 0, 1: {tensor}"

    mapped_tensor = (tensor + 1).to(torch.uint8) # -1->0, 0->1, 1->2

    num_elements = mapped_tensor.numel()
    
    bit_buffer = 0
    bits_in_buffer = 0
    packed_data = bytearray()

    for i in range(num_elements):
        val = mapped_tensor[i].item() # 0, 1, or 2
        bit_buffer |= ((val & 0x7) << bits_in_buffer)
        bits_in_buffer += 3

        while bits_in_buffer >= 8:
            packed_data.append(bit_buffer & 0xFF)
            bit_buffer >>= 8
            bits_in_buffer -= 8

    if bits_in_buffer > 0:
        packed_data.append(bit_buffer & 0xFF)

    return bytes(packed_data)

# --- Main Export Logic ---

def export_qat_unet_to_binary(model_path, output_dir="exported_unet_qat_model"):
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "qat_unet_model.bin")

    # Load the model configuration (must match how your QATUNet was trained)
    model_options = diffusion_defaults()
    model_options['image_size'] = 32
    model_options['in_channels'] = 3
    model_options['num_channels'] = 128 # base_channels
    model_options['num_res_blocks'] = 2
    model_options['num_heads'] = 4
    model_options['num_heads_upsample'] = -1
    model_options['num_head_channels'] = 64
    model_options['attention_resolutions'] = "16,8" # These are resolutions, not channel counts
    model_options['channel_mult'] = "1,2,2,2"
    model_options['dropout'] = 0.1
    model_options['class_cond'] = True
    model_options['use_scale_shift_norm'] = True
    model_options['resblock_updown'] = False
    model_options['use_fp16'] = False
    model_options['initial_pad'] = 0
    model_options['use_new_attention_architecture'] = False

    model_options['time_emb_scale'] = 1.0
    model_options['num_groups'] = 32
    model_options['time_emb_dim'] = model_options['num_channels'] * 4

    model_options['attention_resolutions'] = [int(x) for x in model_options['attention_resolutions'].split(',')]
    model_options['channel_mult'] = [int(x) for x in model_options['channel_mult'].split(',')]
    
    model_options['num_classes'] = 10 if model_options['class_cond'] else 0

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
        norm='gn',
        num_groups=model_options['num_groups'],
        initial_pad=model_options['initial_pad']
    )

    print(f"Loading model from {model_path} with strict=False...")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval() # Set to evaluation mode

    print("Model loaded successfully. Starting export...")

    with open(output_file_path, 'wb') as f:
        # --- Write Model Hyperparameters ---
        f.write(struct.pack('<I', model_options['image_size']))
        f.write(struct.pack('<I', model_options['in_channels']))
        f.write(struct.pack('<I', model_options['num_channels']))
        f.write(struct.pack('<I', len(model_options['channel_mult'])))
        for mult in model_options['channel_mult']:
            f.write(struct.pack('<I', mult))
        f.write(struct.pack('<I', model_options['num_res_blocks']))
        f.write(struct.pack('<I', model_options['time_emb_dim']))
        f.write(struct.pack('<f', model_options['time_emb_scale']))
        f.write(struct.pack('<I', model_options['num_classes']))
        f.write(struct.pack('<f', model_options['dropout']))
        f.write(struct.pack('<I', len(model_options['attention_resolutions'])))
        for res in model_options['attention_resolutions']:
            f.write(struct.pack('<I', res))
        f.write(struct.pack('<I', model_options['num_groups']))
        f.write(struct.pack('<I', model_options['initial_pad']))
        f.write(struct.pack('<?', model_options['use_scale_shift_norm'])) # Boolean as 1 byte

        # --- Helper to write a QATConv2d layer's parameters ---
        def write_qat_conv_layer(layer_name, layer, file_handle):
            print(f"  Exporting QATConv2d layer: {layer_name}")
            # Directly calculate weight_scale from the weight tensor itself
            weight_scale = layer.weight.abs().mean().item()
            # Ternarize the weight tensor directly for packing
            ternary_weights = ternarize_tensor_to_int8(layer.weight)
            packed_weights = pack_ternary_weights_5x3bit(ternary_weights.flatten())

            bias = layer.bias.detach().cpu().numpy() if layer.bias is not None else np.array([])

            print(f"    DEBUG (Python): {layer_name} - weight_scale = {weight_scale:.4f}")
            print(f"    DEBUG (Python): {layer_name} - packed_weights_len = {len(packed_weights)}")
            print(f"    DEBUG (Python): {layer_name} - bias_size = {bias.size}")

            # Write layer metadata (order MUST match C++ read order)
            file_handle.write(struct.pack('<I', layer.in_channels))
            file_handle.write(struct.pack('<I', layer.out_channels))
            file_handle.write(struct.pack('<I', layer.kernel_size[0]))
            file_handle.write(struct.pack('<I', layer.kernel_size[1]))
            file_handle.write(struct.pack('<I', layer.stride[0]))
            file_handle.write(struct.pack('<I', layer.stride[1]))
            file_handle.write(struct.pack('<I', layer.padding[0]))
            file_handle.write(struct.pack('<I', layer.padding[1]))
            file_handle.write(struct.pack('<I', layer.groups))
            file_handle.write(struct.pack('<f', weight_scale))

            # Write packed weights
            file_handle.write(struct.pack('<I', len(packed_weights)))
            file_handle.write(packed_weights)

            # Write bias
            file_handle.write(struct.pack('<I', bias.size))
            file_handle.write(bias.tobytes())

        # --- Helper to write a standard (float) Conv2d layer's parameters ---
        def write_float_conv_layer(layer_name, layer, file_handle):
            print(f"  Exporting Float Conv2d layer: {layer_name}")
            weight = layer.weight.detach().cpu().numpy()
            bias = layer.bias.detach().cpu().numpy() if layer.bias is not None else np.array([])

            print(f"    DEBUG (Python): {layer_name} - weight_shape = {weight.shape}")
            print(f"    DEBUG (Python): {layer_name} - bias_size = {bias.size}")

            # Write layer metadata (order MUST match C++ read order for FloatConv2dLayer)
            file_handle.write(struct.pack('<I', layer.in_channels))
            file_handle.write(struct.pack('<I', layer.out_channels))
            file_handle.write(struct.pack('<I', layer.kernel_size[0]))
            file_handle.write(struct.pack('<I', layer.kernel_size[1]))
            file_handle.write(struct.pack('<I', layer.stride[0]))
            file_handle.write(struct.pack('<I', layer.stride[1]))
            file_handle.write(struct.pack('<I', layer.padding[0]))
            file_handle.write(struct.pack('<I', layer.padding[1]))
            file_handle.write(struct.pack('<I', layer.groups))

            # Write float weights
            file_handle.write(struct.pack('<I', weight.size))
            file_handle.write(weight.tobytes())

            # Write bias
            file_handle.write(struct.pack('<I', bias.size))
            file_handle.write(bias.tobytes())


        def write_groupnorm_layer(layer_name, layer, file_handle):
            print(f"  Exporting GroupNorm: {layer_name}")
            num_groups = layer.num_groups
            num_channels = layer.num_channels
            eps = layer.eps
            gamma = layer.weight.detach().cpu().numpy() if layer.weight is not None else np.array([])
            beta = layer.bias.detach().cpu().numpy() if layer.bias is not None else np.array([])

            print(f"    DEBUG (Python): {layer_name} - num_groups = {num_groups}")
            print(f"    DEBUG (Python): {layer_name} - num_channels = {num_channels}")
            print(f"    DEBUG (Python): {layer_name} - eps = {eps:.10f}")
            print(f"    DEBUG (Python): {layer_name} - gamma_size = {gamma.size}")
            print(f"    DEBUG (Python): {layer_name} - beta_size = {beta.size}")

            file_handle.write(struct.pack('<I', num_groups))
            file_handle.write(struct.pack('<I', num_channels))
            file_handle.write(struct.pack('<f', eps))

            file_handle.write(struct.pack('<I', gamma.size))
            file_handle.write(gamma.tobytes())
            file_handle.write(struct.pack('<I', beta.size))
            file_handle.write(beta.tobytes())


        def write_linear_layer(layer_name, layer, file_handle):
            print(f"  Exporting Linear layer: {layer_name}")
            # For Linear layers, we manually apply ternary quantization and packing
            weight_tensor = layer.weight.detach().cpu()
            weight_scale = weight_tensor.abs().mean().item()
            ternary_weights = ternarize_tensor_to_int8(weight_tensor)
            packed_weights = pack_ternary_weights_5x3bit(ternary_weights.flatten())

            bias = layer.bias.detach().cpu().numpy() if layer.bias is not None else np.array([])

            print(f"    DEBUG (Python): {layer_name} - out_features = {layer.out_features}")
            print(f"    DEBUG (Python): {layer_name} - in_features = {layer.in_features}")
            print(f"    DEBUG (Python): {layer_name} - weight_scale = {weight_scale:.4f}")
            print(f"    DEBUG (Python): {layer_name} - packed_weights_len = {len(packed_weights)}")
            print(f"    DEBUG (Python): {layer_name} - bias_size = {bias.size}")

            # Write layer metadata (order matches C++ read order)
            file_handle.write(struct.pack('<I', layer.out_features))
            file_handle.write(struct.pack('<I', layer.in_features))
            file_handle.write(struct.pack('<f', weight_scale))

            file_handle.write(struct.pack('<I', len(packed_weights)))
            file_handle.write(packed_weights)

            file_handle.write(struct.pack('<I', bias.size))
            file_handle.write(bias.tobytes())

        # --- Helper to write a HadamardTransform layer's parameters ---
        # HadamardTransform does NOT have trainable parameters to export.
        # It generates its matrix dynamically.
        def write_hadamard_transform_layer(layer_name, layer, file_handle):
            print(f"  Skipping HadamardTransform layer: {layer_name} (no trainable parameters to export)")
            # You might want to save its 'dim' or 'target_dim' if C++ needs to reconstruct it
            # For now, assuming C++ reconstructs based on architecture info.
            pass


        # --- Recursive module export ---
        def export_module(name, module, file_handle):
            # Check for QATConv2d (quantized conv)
            if isinstance(module, QATConv2d):
                write_qat_conv_layer(name, module, file_handle)
            # Check for standard nn.Conv2d (like out_conv, unquantized)
            elif isinstance(module, nn.Conv2d): # This must be AFTER QATConv2d check if QATConv2d inherits from nn.Conv2d
                write_float_conv_layer(name, module, file_handle)
            elif isinstance(module, nn.GroupNorm):
                write_groupnorm_layer(name, module, file_handle)
            elif isinstance(module, nn.Linear): # This handles time_mlp and residual_connection's time_bias/class_bias
                write_linear_layer(name, module, file_handle)
            elif isinstance(module, nn.Embedding): # Handle nn.Embedding layers (e.g., class_bias)
                # nn.Embedding has a 'weight' parameter, which is a float tensor.
                # It's not ternary quantized in your QAT_UNet.py.
                # Export it as a float weight matrix.
                print(f"  Exporting Embedding layer: {name}")
                embedding_weight = module.weight.detach().cpu().numpy()
                print(f"    DEBUG (Python): {name} - embedding_weight_shape = {embedding_weight.shape}")
                
                file_handle.write(struct.pack('<I', module.num_embeddings)) # num_classes
                file_handle.write(struct.pack('<I', module.embedding_dim)) # embedding_dim
                file_handle.write(struct.pack('<I', embedding_weight.size)) # total number of floats
                file_handle.write(embedding_weight.tobytes())
            elif isinstance(module, HadamardTransform):
                write_hadamard_transform_layer(name, module, file_handle)
            elif isinstance(module, (nn.ModuleList, nn.Sequential, AttentionBlock)): # Recurse into these containers
                for child_name, child_module in module.named_children():
                    export_module(f"{name}.{child_name}", child_module, file_handle)
            elif isinstance(module, (nn.Identity, ScaledActivationTernary, ScaledWeightTernary, PositionalEmbedding, nn.Dropout, nn.Upsample, nn.SiLU, nn.MaxPool2d)):
                # These modules do not have trainable parameters to export or are handled by their parent
                print(f"  Skipping non-parameter module: {name} ({type(module).__name__})")
                pass
            else:
                # Default behavior: recurse into children if it's a generic nn.Module
                # This catches QATResidualBlock, Downsample, Upsample which are custom Modules
                # but their children are handled by the specific checks above.
                has_children = False
                for child_name, child_module in module.named_children():
                    has_children = True
                    export_module(f"{name}.{child_name}", child_module, file_handle)
                if not has_children and len(list(module.parameters(recurse=False))) > 0:
                    print(f"  WARNING: Module '{name}' of type '{type(module).__name__}' has parameters but no specific export logic. Skipping parameters.")


        # Start exporting from the top-level model
        export_module("model", model, f)

    print(f"Model exported to {output_file_path}")
    print("Please note: This script is a starting point. You'll need to meticulously match")
    print("the C++ loading logic with the exact order and format of parameters saved here.")
    print("Complexities like skip connections, time embeddings, and class conditioning")
    print("will require careful C++ implementation mirroring the PyTorch model's forward pass.")


if __name__ == "__main__":
    TRAINED_MODEL_PATH = "qat_unet_progressive.pth" # Path to your trained QAT UNet model

    if not os.path.exists(TRAINED_MODEL_PATH):
        print(f"Error: Trained model not found at {TRAINED_MODEL_PATH}")
        print("Please update TRAINED_MODEL_PATH to your actual checkpoint file.")
    else:
        export_qat_unet_to_binary(TRAINED_MODEL_PATH)
