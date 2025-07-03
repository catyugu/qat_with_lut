import torch
import struct
import argparse
from ddpm.script_utils_qat import create_qat_model
from ddpm.QAT_UNet import QuantizedConv2d, QuantizedLinear

def pack_weights(weights_int8):
    """Packs 5x 3-bit int8 weights into 2 bytes (uint8_t)."""
    # This must exactly match the packing logic your C++ kernel expects.
    # This is a placeholder for your specific 5x3-bit packing scheme.
    # The actual implementation is complex and hardware-specific.
    print("Warning: pack_weights is a placeholder and needs a real implementation.")
    # Note: The data needs to be flattened before packing if it's not already.
    flat_weights = weights_int8.flatten().numpy()
    return flat_weights.astype('uint8')


def export_unet_to_binary(model_path: str, output_path: str, args):
    """
    Loads a trained QAT U-Net model and exports its weights and parameters
    to a custom binary file for the C++ application.
    """
    print(f"Loading model from: {model_path}")
    
    # 1. Create the model object directly using create_qat_model.
    # This is more direct than get_models as we don't need the diffusion object.
    # Note the use of eval() on channel_mult, which is required by create_qat_model.
    model = create_qat_model(
        image_size=args.image_size,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        learn_sigma=args.learn_sigma,
        class_cond=args.class_cond,
        use_checkpoint=False,  # This is not configured in the training script args
        attention_resolutions=args.attention_resolutions,
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        channel_mult=eval(args.channel_mult),
        use_scale_shift_norm=args.use_scale_shift_norm,
        dropout=args.dropout,
        bit=args.bit
    )
    
    # 2. Load the state dictionary into the model (the U-Net)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    unet = model # The model object is the U-Net we need to export
    unet.eval()

    print("Model loaded successfully. Starting export...")

    conv_layers = {name: mod for name, mod in unet.named_modules() if isinstance(mod, QuantizedConv2d)}
    linear_layers = {name: mod for name, mod in unet.named_modules() if isinstance(mod, QuantizedLinear)}

    with open(output_path, "wb") as f:
        # --- Header ---
        # Write some metadata first
        f.write(struct.pack('i', unet.model_channels))
        f.write(struct.pack('i', unet.num_res_blocks))
        f.write(struct.pack('I', len(conv_layers)))
        f.write(struct.pack('I', len(linear_layers)))

        # --- Convolutional Layers ---
        print(f"Exporting {len(conv_layers)} convolutional layers...")
        for name, layer in conv_layers.items():
            # Name
            f.write(struct.pack('I', len(name)))
            f.write(name.encode('utf-8'))

            # Layer parameters
            f.write(struct.pack('i', layer.in_channels))
            f.write(struct.pack('i', layer.out_channels))
            f.write(struct.pack('i', layer.kernel_size[0]))
            f.write(struct.pack('i', layer.stride[0]))
            f.write(struct.pack('i', layer.padding[0]))
            f.write(struct.pack('f', layer.activation_scale.item()))

            # Weights (quantized and packed)
            weights_quant = layer.weight_quant(layer.weight).detach()
            weights_packed = pack_weights(weights_quant) # You need to implement this!
            f.write(struct.pack('Q', weights_packed.size))
            f.write(weights_packed.tobytes())
            
            # Bias
            bias = layer.bias.detach().numpy()
            f.write(struct.pack('Q', bias.size))
            f.write(bias.tobytes())

        # --- Linear Layers ---
        print(f"Exporting {len(linear_layers)} linear layers...")
        for name, layer in linear_layers.items():
            # Name
            f.write(struct.pack('I', len(name)))
            f.write(name.encode('utf-8'))
            
            # Layer parameters
            f.write(struct.pack('i', layer.in_features))
            f.write(struct.pack('i', layer.out_features))
            f.write(struct.pack('f', layer.activation_scale.item()))
            
            # Weights
            weights_quant = layer.weight_quant(layer.weight).detach()
            weights_packed = pack_weights(weights_quant)
            f.write(struct.pack('Q', weights_packed.size))
            f.write(weights_packed.tobytes())
            
            # Bias
            bias = layer.bias.detach().numpy()
            f.write(struct.pack('Q', bias.size))
            f.write(bias.tobytes())


    print(f"Successfully exported model to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export QAT U-Net to C++ binary format.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pt model file.")
    parser.add_argument("--output_path", type=str, default="qat_unet.bin", help="Path to save the output binary file.")
    
    # Add all the arguments that get_models expects, with defaults from train_qat_cifar.py
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--num_channels', type=int, default=128)
    parser.add_argument('--num_res_blocks', type=int, default=3)
    parser.add_argument('--learn_sigma', type=bool, default=True)
    parser.add_argument('--class_cond', type=bool, default=False)
    parser.add_argument('--attention_resolutions', type=str, default='16')
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_head_channels', type=int, default=-1)
    parser.add_argument('--channel_mult', type=str, default='(1,2,2,2)')
    parser.add_argument('--use_scale_shift_norm', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--bit', type=int, default=3)
    # These args are for the diffusion process and not strictly needed for model creation,
    # but we include them for completeness with the training script's args.
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--noise_schedule', type=str, default='linear')
    parser.add_argument('--timestep_respacing', type=str, default='')
    parser.add_argument('--use_kl', type=bool, default=False)
    parser.add_argument('--predict_xstart', type=bool, default=False)
    parser.add_argument('--rescale_timesteps', type=bool, default=True)
    parser.add_argument('--rescale_learned_sigmas', type=bool, default=True)

    args = parser.parse_args()

    export_unet_to_binary(args.model_path, args.output_path, args)
