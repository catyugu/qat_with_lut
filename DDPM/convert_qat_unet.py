import torch
import torch.nn as nn
import struct
import numpy as np
from ddpm.QAT_UNet import QATUNet, QATResidualBlock, AttentionBlock, Downsample, Upsample, QATConv2d, ScaledWeightTernary
import os
# --- Helper Functions ---

# MODIFIED: Added alpha_val parameter
def pack_ternary_weights(tensor, alpha_val):
    """
    将一个三值化的权重张量 (-alpha, 0, alpha) 打包成一个 uint32_t 列表。
    每个权重使用2个bit: 01 for +1 (scaled by alpha), 10 for -1 (scaled by alpha), 00 for 0.
    """
    tensor_flat = tensor.cpu().view(-1)
    
    packed_data = []
    num_chunks = (len(tensor_flat) + 15) // 16 # 每 16 个权重打包成一个 uint32
    
    # Define a small tolerance for floating-point comparisons
    tolerance = 1e-6 # Weights might not be *exactly* alpha or -alpha due to float precision

    for i in range(num_chunks):
        packed_uint32 = 0
        chunk = tensor_flat[i*16 : (i+1)*16]
        
        for j, weight_val_scaled in enumerate(chunk):
            bits = 0
            
            # CORRECTED LOGIC: Compare with scaled alpha values
            # The original 'weight.item()' in the loop was {-alpha, 0, +alpha}
            # We need to determine if it conceptually represents -1, 0, or +1
            if abs(weight_val_scaled.item() - alpha_val) < tolerance:
                bits = 1  # 01 for +1 (i.e., +alpha)
            elif abs(weight_val_scaled.item() + alpha_val) < tolerance:
                bits = 2  # 10 for -1 (i.e., -alpha)
            # If it's effectively 0, bits remains 0 (00)
            
            # 将2-bit的数据移到正确的位置并合并 (LSB to MSB packing)
            packed_uint32 |= (bits << (j * 2))
            
        packed_data.append(packed_uint32)
        
    return packed_data

# Helper functions to write data to the binary file (unchanged)
def write_int(f, val):
    f.write(struct.pack('i', val))

def write_float(f, val):
    f.write(struct.pack('f', val))

def write_bool(f, val):
    f.write(struct.pack('?', val))

def write_tensor(f, tensor):
    tensor_np = tensor.detach().cpu().numpy()
    f.write(tensor_np.tobytes())

# --- Layer Export Functions ---

def export_conv(f, layer):
    """
    Exports a Conv2d layer.
    For QATConv2d, weights are ternary quantized and packed into 2-bit format.
    For standard nn.Conv2d, weights are exported as floats.
    """
    is_custom_qat = isinstance(layer, QATConv2d)

    # Write common parameters
    write_int(f, layer.in_channels)
    write_int(f, layer.out_channels)
    write_int(f, layer.kernel_size[0])
    write_int(f, layer.kernel_size[1])
    write_int(f, layer.stride[0])
    write_int(f, layer.stride[1])
    write_int(f, layer.padding[0])
    write_int(f, layer.padding[1])
    write_int(f, layer.groups)

    if is_custom_qat:
        print(f"  Exporting and Packing Ternary QAT Conv2d: in={layer.in_channels}, out={layer.out_channels}")
        weight = layer.weight.detach()
        bias = layer.bias.detach()

        # Apply the custom ternary quantization to get {-alpha, 0, +alpha} float tensor
        quantized_weight_float = ScaledWeightTernary.apply(weight)

        # 1. Write scale (alpha)
        alpha = torch.mean(torch.abs(weight)).item()
        write_float(f, alpha)

        # 2. Pack the weights into a list of uint32_t
        # MODIFIED: Pass alpha to pack_ternary_weights
        packed_data_uint32 = pack_ternary_weights(quantized_weight_float, alpha)
        
        # 3. Write the original tensor shape for C++ to reconstruct the layout
        shape = weight.shape
        write_int(f, len(shape)) # number of dimensions
        for dim_size in shape:
            write_int(f, dim_size)

        # 4. Write the packed data itself
        write_int(f, len(packed_data_uint32)) # number of uint32_t values
        for val in packed_data_uint32:
            f.write(struct.pack('I', val)) # 'I' for unsigned int (32-bit)

        # 5. Write bias (unchanged)
        write_int(f, bias.numel())
        write_tensor(f, bias)
        # Debug prints for model.conv_in (keep for verification after fix)
        if layer.in_channels == 3 and layer.out_channels == 128 and layer.kernel_size == (3, 3):
            print(f"\n--- Debugging quantized_weight_float for model.conv_in ---")
            print(f"Shape: {quantized_weight_float.shape}")
            print(f"Mean: {quantized_weight_float.mean().item()}")
            print(f"Std Dev: {quantized_weight_float.std().item()}")
            print(f"Min: {quantized_weight_float.min().item()}")
            print(f"Max: {quantized_weight_float.max().item()}")
            print(f"First 100 quantized_weight_float values (flat): {quantized_weight_float.view(-1)[:100].tolist()}")
            # Add print for packed_data_uint32 after packing to confirm non-zeros
            print(f"First 5 packed_data_uint32 values: {packed_data_uint32[:5]}")
            print(f"--- End Debugging model.conv_in ---")

    else: # Standard nn.Conv2d (unchanged)
        print(f"  Exporting Float Conv2d: in={layer.in_channels}, out={layer.out_channels}")
        weight, bias = layer.weight.detach(), layer.bias.detach()
        
        write_float(f, 1.0) # alpha = 1.0 for float models
        
        shape = weight.shape
        write_int(f, len(shape))
        for dim_size in shape:
            write_int(f, dim_size)

        write_int(f, -1) # Use -1 as a flag for "unpacked float data"
        write_int(f, weight.numel())
        write_tensor(f, weight)
        
        write_int(f, bias.numel())
        write_tensor(f, bias)

# --- Other Export Functions (unchanged) ---
def export_linear(f, layer):
    print(f"  Exporting Float Linear: in={layer.in_features}, out={layer.out_features}")
    weight, bias = layer.weight, layer.bias

    write_int(f, layer.in_features)
    write_int(f, layer.out_features)

    write_int(f, weight.numel())
    write_tensor(f, weight)

    write_int(f, bias.numel())
    write_tensor(f, bias)

def export_embedding(f, layer):
    print(f"  Exporting Embedding: num_embeddings={layer.num_embeddings}, embedding_dim={layer.embedding_dim}")
    write_int(f, layer.num_embeddings)
    write_int(f, layer.embedding_dim)
    write_int(f, layer.weight.numel())
    write_tensor(f, layer.weight)

def export_norm(f, layer):
    write_int(f, layer.num_groups)
    write_int(f, layer.num_channels)
    write_float(f, layer.eps)
    write_int(f, layer.weight.numel())
    write_tensor(f, layer.weight)
    write_int(f, layer.bias.numel())
    write_tensor(f, layer.bias)
    print(f"  Exported GroupNorm: groups={layer.num_groups}, channels={layer.num_channels}")

def export_attention_block(f, block):
    print("  Exporting Nested AttentionBlock...")
    export_norm(f, block.norm)
    export_conv(f, block.to_qkv)
    export_conv(f, block.to_out)

def export_res_block(f, block):
    print("Exporting QATResidualBlock...")
    export_norm(f, block.norm_1)
    export_conv(f, block.conv_1)

    has_time_bias = block.time_bias is not None
    write_bool(f, has_time_bias)
    if has_time_bias:
        export_linear(f, block.time_bias)

    has_class_bias = block.class_bias is not None
    write_bool(f, has_class_bias)
    if has_class_bias:
        export_embedding(f, block.class_bias)

    export_norm(f, block.norm_2)
    export_conv(f, block.conv_2[1])

    is_res_conv = isinstance(block.residual_connection, nn.Conv2d)
    write_bool(f, is_res_conv)
    if is_res_conv:
        export_conv(f, block.residual_connection)

    has_attention = isinstance(block.attention, AttentionBlock)
    write_bool(f, has_attention)
    if has_attention:
        export_attention_block(f, block.attention)

# --- Main Export Logic (unchanged) ---

def main():
    config = {
        "image_size": 32,
        "in_channels": 3,
        "base_channels": 128,
        "channel_mults": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "time_emb_dim": 512,
        "time_emb_scale": 1.0,
        "num_classes": 10,
        "dropout": 0.1,
        "attention_resolutions": [1],
        "norm": "gn",
        "num_groups": 32,
        "initial_pad": 0
    }

    model_path = "qat_unet_final.pth"
    output_path = "qat_unet_model_packed.bin"

    model_init_config = {
        "img_channels": config["in_channels"],
        "base_channels": config["base_channels"],
        "channel_mults": config["channel_mults"],
        "num_res_blocks": config["num_res_blocks"],
        "time_emb_dim": config["time_emb_dim"],
        "time_emb_scale": config["time_emb_scale"],
        "num_classes": config["num_classes"],
        "dropout": config["dropout"],
        "attention_resolutions": config["attention_resolutions"],
        "norm": config["norm"],
        "num_groups": config["num_groups"],
        "initial_pad": config["initial_pad"]
    }

    print("Loading QAT UNet model...")
    model = QATUNet(**model_init_config)

    # Check if the EMA model path exists
    if not os.path.exists(model_path):
        print(f"Error: EMA model not found at {model_path}. Please ensure your training script saves 'ema_model.pth'.")
        return

    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    model.eval()
    print("Model loaded successfully.")

    with open(output_path, "wb") as f:
        print("\n--- Starting Export ---")

        # 1. Write Hyperparameters
        print("Writing hyperparameters...")
        write_int(f, config["image_size"])
        write_int(f, config["in_channels"])
        write_int(f, config["base_channels"])
        write_int(f, len(config["channel_mults"]))
        for mult in config["channel_mults"]:
            write_int(f, mult)
        write_int(f, config["num_res_blocks"])
        write_int(f, config["time_emb_dim"])
        write_float(f, config["time_emb_scale"])
        write_int(f, 0 if config["num_classes"] is None else config["num_classes"])
        write_float(f, config["dropout"])
        write_int(f, len(config["attention_resolutions"]))
        for res in config["attention_resolutions"]:
            write_int(f, res)
        write_int(f, config["num_groups"])
        write_int(f, config["initial_pad"])
        write_bool(f, True) # use_scale_shift_norm

        # 2. Export Layers
        print("\nExporting Time MLP...")
        export_linear(f, model.time_mlp[1])
        export_linear(f, model.time_mlp[3])

        print("\nExporting Initial Convolution...")
        export_conv(f, model.init_conv)

        print("\nExporting Downsampling Path...")
        for layer in model.downs:
            if isinstance(layer, QATResidualBlock):
                export_res_block(f, layer)
            elif isinstance(layer, Downsample):
                export_conv(f, layer.downsample)

        print("\nExporting Middle Path...")
        for layer in model.mid:
             if isinstance(layer, QATResidualBlock):
                export_res_block(f, layer)

        print("\nExporting Upsampling Path...")
        for layer in model.ups:
            if isinstance(layer, QATResidualBlock):
                export_res_block(f, layer)
            elif isinstance(layer, Upsample):
                export_conv(f, layer.upsample[1])

        print("\nExporting Final Layers...")
        export_norm(f, model.out_norm)
        export_conv(f, model.out_conv)

        print("\n--- Export Complete ---")
        print(f"Model successfully exported to {output_path}")

if __name__ == "__main__":
    main()