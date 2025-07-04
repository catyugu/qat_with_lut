# DDPM/ddpm/QAT_UNet.py
# Final version with a robust, scaled activation quantizer for stable training.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =================================================================================
# 1. Ternary Quantization Components
# =================================================================================

class ScaledWeightTernary(torch.autograd.Function):
    """
    A robust ternary weight quantizer that uses a layer-wise scaling factor.
    """
    @staticmethod
    def forward(ctx, weight):
        alpha = torch.mean(torch.abs(weight)).detach()
        threshold = 0.001 * alpha
        quantized_weight = torch.where(weight > threshold, 1.0, 
                                       torch.where(weight < -threshold, -1.0, 0.0))
        scaled_quantized_weight = alpha * quantized_weight
        ctx.save_for_backward(weight)
        return scaled_quantized_weight

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        grad_weight = grad_output.clone()
        grad_weight[weight.abs() > 1.0] = 0 
        return grad_weight

class ScaledActivationTernary(torch.autograd.Function):
    """
    A robust ternary activation quantizer that calculates a scaling factor
    on-the-fly from the input tensor, similar to the weight quantizer.
    """
    @staticmethod
    def forward(ctx, x):
        alpha = torch.mean(torch.abs(x)).detach()
        threshold = 0.001 * alpha
        quantized_x = torch.where(x > threshold, 1.0, 
                                  torch.where(x < -threshold, -1.0, 0.0))
        scaled_quantized_x = alpha * quantized_x
        ctx.save_for_backward(x)
        return scaled_quantized_x

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[x.abs() > 1.0] = 0
        return grad_x

class QATConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantize_weights = True

    def forward(self, x):
        if self.quantize_weights:
            quantized_weight = ScaledWeightTernary.apply(self.weight)
            return F.conv2d(self, x, quantized_weight, self.bias)
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

# =================================================================================
# 2. Robust UNet Structure
# =================================================================================

def get_norm(norm, num_channels, num_groups):
    if norm == "gn": return nn.GroupNorm(num_groups, num_channels)
    else: return nn.Identity()

class PositionalEmbedding(nn.Module):
    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.downsample = QATConv2d(in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x, time_emb, y):
        return self.downsample(x)

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            QATConv2d(in_channels, in_channels, 3, padding=1),
        )

    def forward(self, x, time_emb, y):
        return self.upsample(x)

class HadamardTransform(nn.Module):
    """
    Applies a Hadamard transform to the input tensor along the last dimension.
    Pads to the nearest power of 2 if necessary.
    Note: For large dimensions, a true Fast Hadamard Transform (FHT) algorithm
    is required for efficiency. This implementation uses direct matrix multiplication
    which can be slow for large N.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Calculate the next power of 2 for padding
        self.target_dim = 2**math.ceil(math.log2(dim)) if dim > 0 else 0
        
        if self.target_dim == 0:
            self.hadamard_matrix = None
        else:
            # Generate the Hadamard matrix
            # Normalized by sqrt(N) for orthogonality
            self.hadamard_matrix = self._get_hadamard_matrix(self.target_dim) / math.sqrt(self.target_dim)

    def _get_hadamard_matrix(self, n):
        # Recursively generates a Hadamard matrix of size n (must be power of 2)
        if n == 1:
            return torch.tensor([[1.]])
        else:
            h_n_div_2 = self._get_hadamard_matrix(n // 2)
            top_row = torch.cat([h_n_div_2, h_n_div_2], dim=1)
            bottom_row = torch.cat([h_n_div_2, -h_n_div_2], dim=1)
            return torch.cat([top_row, bottom_row], dim=0)

    def forward(self, x):
        if self.hadamard_matrix is None:
            # If dim is 0 or invalid, return original or handle error
            return x 
        
        original_shape = x.shape # (B, C, H, W)
        # Reshape to (Batch*H*W, Channels) for matrix multiplication
        # We want to apply Hadamard along the channel dimension (dim=1 in original x)
        x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, self.dim) # (B*H*W, C)

        # Pad if the original channel dimension is not a power of 2
        if self.dim < self.target_dim:
            padding_needed = self.target_dim - self.dim
            x_padded = F.pad(x_reshaped, (0, padding_needed)) # Pad last dimension
        else:
            x_padded = x_reshaped

        # Move Hadamard matrix to the correct device
        hadamard_matrix_on_device = self.hadamard_matrix.to(x.device)
        
        # Apply Hadamard transform (matrix multiplication)
        transformed_x = torch.matmul(x_padded, hadamard_matrix_on_device)

        # Unpad if necessary
        if self.dim < self.target_dim:
            transformed_x = transformed_x[:, :self.dim] # Slice back to original dimension
        
        # Reshape back to (B, C, H, W)
        transformed_x = transformed_x.reshape(original_shape[0], original_shape[2], original_shape[3], original_shape[1]).permute(0, 3, 1, 2)
        
        return transformed_x


class AttentionBlock(nn.Module): # Renamed from QATAttentionBlock
    def __init__(self, in_channels, norm="gn", num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        self.norm = get_norm(norm, in_channels, num_groups)
        # CHANGED: Use QATConv2d for quantized linear layers
        self.to_qkv = QATConv2d(in_channels, in_channels * 3, 1)
        self.to_out = QATConv2d(in_channels, in_channels, 1)

        # Initialize Hadamard transforms for Q, K, V
        self.hadamard_q = HadamardTransform(in_channels)
        self.hadamard_k = HadamardTransform(in_channels)
        self.hadamard_v = HadamardTransform(in_channels)


    def forward(self, x):
        b, c, h, w = x.shape
        # Project to Q, K, V using quantized convolutions
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)
        
        # Apply Hadamard Transform to Q, K, V (operates on float outputs of QATConv2d)
        q = self.hadamard_q(q)
        k = self.hadamard_k(k)
        v = self.hadamard_v(v)

        # Reshape for attention
        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)
        
        # Scaled Dot-Product Attention (full precision)
        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        attention = torch.softmax(dot_products, dim=-1)
        out = torch.bmm(attention, v)
        
        # Reshape back and final projection using quantized convolution
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)
        return self.to_out(out) + x

class QATResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, time_emb_dim, num_classes, norm, num_groups, use_attention):
        super().__init__()
        self.quantize_activations = True
        self.full_precision_activation = nn.SiLU()
        self.time_activation = nn.SiLU()
        
        self.norm_1 = get_norm(norm, in_channels, num_groups)
        self.conv_1 = QATConv2d(in_channels, out_channels, 3, padding=1)
        self.norm_2 = get_norm(norm, out_channels, num_groups)
        self.conv_2 = nn.Sequential(nn.Dropout(p=dropout), QATConv2d(out_channels, out_channels, 3, padding=1))
        self.time_bias = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None
        self.class_bias = nn.Embedding(num_classes, out_channels) if num_classes is not None else None
        self.residual_connection = QATConv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.attention = nn.Identity() if not use_attention else AttentionBlock(out_channels, norm, num_groups) # Renamed

    def forward(self, x, time_emb=None, y=None):
        h = self.norm_1(x)
        
        if self.quantize_activations and self.training:
            h = ScaledActivationTernary.apply(h)
        else:
            h = self.full_precision_activation(h)
            
        h = self.conv_1(h)
        
        if self.time_bias is not None:
            h += self.time_bias(self.time_activation(time_emb))[:, :, None, None]
        if self.class_bias is not None:
            h += self.class_bias(y)[:, :, None, None]
            
        h2 = self.norm_2(h)
        
        if self.quantize_activations and self.training:
            h2 = ScaledActivationTernary.apply(h2)
        else:
            h2 = self.full_precision_activation(h2)
            
        h2 = self.conv_2(h2)
        
        return self.attention(h2 + self.residual_connection(x))

class QATUNet(nn.Module):
    def __init__(self, img_channels, base_channels, channel_mults, num_res_blocks, time_emb_dim, time_emb_scale, num_classes, dropout, attention_resolutions, norm, num_groups, initial_pad):
        super().__init__()
        self.initial_pad = initial_pad
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale),
            nn.Linear(base_channels, time_emb_dim), nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        ) if time_emb_dim is not None else None
        self.init_conv = QATConv2d(img_channels, base_channels, 3, padding=1)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels
        
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                use_attention = (now_channels in attention_resolutions)
                self.downs.append(QATResidualBlock(now_channels, out_channels, dropout, time_emb_dim, num_classes, norm, num_groups, use_attention))
                now_channels = out_channels
                channels.append(now_channels)
            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)
        
        self.mid = nn.ModuleList([
            QATResidualBlock(now_channels, now_channels, dropout, time_emb_dim, num_classes, norm, num_groups, True),
            QATResidualBlock(now_channels, now_channels, dropout, time_emb_dim, num_classes, norm, num_groups, False),
        ])
        
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks + 1):
                use_attention = (now_channels in attention_resolutions)
                self.ups.append(QATResidualBlock(channels.pop() + now_channels, out_channels, dropout, time_emb_dim, num_classes, norm, num_groups, use_attention))
                now_channels = out_channels
            if i != 0:
                self.ups.append(Upsample(now_channels))
        
        self.out_norm = get_norm(norm, base_channels, num_groups)
        self.out_activation = nn.SiLU()
        self.out_conv = QATConv2d(base_channels, img_channels, 3, padding=1)

    def forward(self, x, time=None, y=None):
        ip = self.initial_pad
        if ip != 0: x = F.pad(x, (ip,) * 4)
        time_emb = self.time_mlp(time) if self.time_mlp is not None else None
        x = self.init_conv(x)
        skips = [x]
        for layer in self.downs:
            x = layer(x, time_emb, y)
            skips.append(x)
        for layer in self.mid:
            x = layer(x, time_emb, y)
        for layer in self.ups:
            if isinstance(layer, QATResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, time_emb, y)
        x = self.out_activation(self.out_norm(x))
        x = self.out_conv(x)
        if ip != 0: return x[:, :, ip:-ip, ip:-ip]
        return x

    def set_quantize_weights(self, enabled: bool):
        for module in self.modules():
            if isinstance(module, QATConv2d):
                module.quantize_weights = enabled
    
    def set_quantize_activations(self, enabled: bool):
        for module in self.modules():
            if isinstance(module, QATResidualBlock):
                module.quantize_activations = enabled
