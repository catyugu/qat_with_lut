import torch
import torch.nn as nn
import struct
import numpy as np
from ddpm.QAT_UNet import QATUNet, QATResidualBlock, AttentionBlock, Downsample, Upsample, QATConv2d, ScaledWeightTernary

# Helper functions to write data to the binary file
def write_int(f, val):
    f.write(struct.pack('i', val))

def write_float(f, val):
    f.write(struct.pack('f', val))

def write_bool(f, val):
    f.write(struct.pack('?', val))

def write_tensor(f, tensor):
    tensor_np = tensor.detach().cpu().numpy()
    f.write(tensor_np.tobytes())

def write_packed_tensor(f, tensor):
    tensor_np = tensor.detach().cpu().numpy()
    f.write(tensor_np.tobytes())

# --- Layer Export Functions ---

def export_conv(f, layer):
    """Exports a Conv2d layer, using custom ternary quantization for QATConv2d."""
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
        print(f"  Exporting Custom Ternary QAT Conv2d: in={layer.in_channels}, out={layer.out_channels}")
        weight = layer.weight.detach()
        bias = layer.bias.detach()

        # Apply the custom ternary quantization logic from QAT_UNet.py
        quantized_weight_float = ScaledWeightTernary.apply(weight)

        # Pack the {-1, 0, 1} float tensor into an int8 tensor for export
        packed_weight_int8 = quantized_weight_float.to(torch.int8)

        # Write scale (alpha) and packed weights
        alpha = torch.mean(torch.abs(weight)).item()
        write_float(f, alpha)
        write_int(f, packed_weight_int8.numel())
        write_packed_tensor(f, packed_weight_int8)

        # Write bias
        write_int(f, bias.numel())
        write_tensor(f, bias)
    else: # Standard nn.Conv2d
        print(f"  Exporting Float Conv2d: in={layer.in_channels}, out={layer.out_channels}")
        weight, bias = layer.weight, layer.bias
        write_float(f, 1.0) # Scale is 1.0 for non-quantized
        write_int(f, weight.numel())
        write_tensor(f, weight)
        write_int(f, bias.numel())
        write_tensor(f, bias)


def export_linear(f, layer):
    """Exports a Linear layer. Assumes it's always float for this model."""
    print(f"  Exporting Float Linear: in={layer.in_features}, out={layer.out_features}")
    weight, bias = layer.weight, layer.bias

    write_int(f, layer.in_features)
    write_int(f, layer.out_features)

    # Export float weights
    write_float(f, 1.0) # Scale is 1.0 for non-quantized
    write_int(f, weight.numel())
    write_tensor(f, weight)

    # Export bias
    write_int(f, bias.numel())
    write_tensor(f, bias)


def export_norm(f, layer):
    """Exports a GroupNorm layer."""
    write_int(f, layer.num_groups)
    write_int(f, layer.num_channels)
    write_float(f, layer.eps)
    write_int(f, layer.weight.numel())
    write_tensor(f, layer.weight)
    write_int(f, layer.bias.numel())
    write_tensor(f, layer.bias)
    print(f"  Exported GroupNorm: groups={layer.num_groups}, channels={layer.num_channels}")

def export_attention_block(f, block):
    """Exports an AttentionBlock."""
    print("  Exporting Nested AttentionBlock...")
    export_norm(f, block.norm)
    export_conv(f, block.to_qkv)
    export_conv(f, block.to_out)

def export_res_block(f, block):
    """Exports a QATResidualBlock based on the provided definition."""
    print("Exporting QATResidualBlock...")
    export_norm(f, block.norm_1)
    export_conv(f, block.conv_1)
    if block.time_bias is not None:
        export_linear(f, block.time_bias)
    export_norm(f, block.norm_2)
    export_conv(f, block.conv_2[1]) # Access the QATConv2d within the nn.Sequential
    if isinstance(block.residual_connection, nn.Conv2d):
        export_conv(f, block.residual_connection)
    if isinstance(block.attention, AttentionBlock):
        export_attention_block(f, block.attention)

# --- Main Export Logic ---

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

    model_path = "qat_unet_progressive.pth"
    output_path = "qat_unet_model.bin"

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