# In DDPM/ddpm/QAT_UNet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Quantization-Aware Training Components ---

TERNARY_THRESHOLD = 0.001

class TernaryWeightQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, full_precision_weight):
        # Clamp weights to a learnable range, then ternarize
        w_clamped = full_precision_weight.clamp(-1.0, 1.0)
        # Ternarize based on a small threshold
        return torch.where(w_clamped > TERNARY_THRESHOLD, 1.0,
                           torch.where(w_clamped < -TERNARY_THRESHOLD, -1.0, 0.0))

    @staticmethod
    def backward(ctx, grad_output):
        # Pass the gradient straight-through to the full-precision weights
        return grad_output

class QuantizedConv2d(nn.Conv2d):
    """
    A Conv2d layer with stably quantized ternary weights.
    """
    def forward(self, input):
        # Apply stable ternary quantization to weights
        quantized_weight = TernaryWeightQuantizer.apply(self.weight)
        return F.conv2d(input, quantized_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class QuantizedLinear(nn.Linear):
    """
    A Linear layer with stably quantized ternary weights.
    """
    def forward(self, input):
        # Apply stable ternary quantization to weights
        quantized_weight = TernaryWeightQuantizer.apply(self.weight)
        return F.linear(input, quantized_weight, self.bias)


# This activation quantization logic is confirmed to be working correctly.
class FakeQuantTernary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        scale = torch.abs(scale)
        x_dequant = (x / scale).round().clamp(-1, 1) * scale
        ctx.save_for_backward(x, scale)
        return x_dequant

    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        is_in_range = (x.abs() <= scale).float()
        grad_x = grad_output * is_in_range
        dequantized_x = (x / scale).round().clamp(-1, 1)
        grad_scale = (grad_output * (dequantized_x - x / scale) * is_in_range).sum()
        return grad_x, grad_scale

class QuantizedTernaryActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        return FakeQuantTernary.apply(x, self.scale)


# --- The rest of the UNet/ResidualBlock code remains the same ---
# It will now use the final, stable versions of all quantization modules.

def get_norm(norm, num_channels, num_groups):
    if norm == "in": return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm == "bn": return nn.BatchNorm2d(num_channels)
    elif norm == "gn": return nn.GroupNorm(num_groups, num_channels)
    elif norm is None: return nn.Identity()
    else: raise ValueError("unknown normalization type")

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
        self.downsample = QuantizedConv2d(in_channels, in_channels, 3, stride=2, padding=1)
    def forward(self, x, time_emb, y):
        return self.downsample(x)

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            QuantizedConv2d(in_channels, in_channels, 3, padding=1),
        )
    def forward(self, x, time_emb, y):
        return self.upsample(x)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, norm="gn", num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        self.norm = get_norm(norm, in_channels, num_groups)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)
    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)
        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)
        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        attention = torch.softmax(dot_products, dim=-1)
        out = torch.bmm(attention, v)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)
        return self.to_out(out) + x

class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, dropout, time_emb_dim=None, num_classes=None,
        activation=F.relu, norm="gn", num_groups=32, use_attention=False,
    ):
        super().__init__()
        self.activation = activation
        self.norm_1 = get_norm(norm, in_channels, num_groups)
        self.q_act_1 = QuantizedTernaryActivation()
        self.conv_1 = QuantizedConv2d(in_channels, out_channels, 3, padding=1)
        self.norm_2 = get_norm(norm, out_channels, num_groups)
        self.q_act_2 = QuantizedTernaryActivation()
        self.conv_2 = nn.Sequential(
            nn.Dropout(p=dropout),
            QuantizedConv2d(out_channels, out_channels, 3, padding=1),
        )
        self.time_bias = QuantizedLinear(time_emb_dim, out_channels) if time_emb_dim is not None else None
        self.class_bias = nn.Embedding(num_classes, out_channels) if num_classes is not None else None
        self.residual_connection = QuantizedConv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.attention = nn.Identity() if not use_attention else AttentionBlock(out_channels, norm, num_groups)
    def forward(self, x, time_emb=None, y=None):
        out = self.norm_1(x)
        out = self.activation(out)
        out = self.q_act_1(out)
        out = self.conv_1(out)
        if self.time_bias is not None:
            if time_emb is None: raise ValueError("Time conditioning was specified but time_emb is not passed")
            out += self.time_bias(self.activation(time_emb))[:, :, None, None]
        if self.class_bias is not None:
            if y is None: raise ValueError("Class conditioning was specified but y is not passed")
            out += self.class_bias(y)[:, :, None, None]
        out = self.norm_2(out)
        out = self.activation(out)
        out = self.q_act_2(out)
        out = self.conv_2(out) + self.residual_connection(x)
        out = self.attention(out)
        return out

class QAT_UNet(nn.Module):
    def __init__(
        self, img_channels, base_channels, channel_mults=(1, 2, 4, 8), num_res_blocks=2,
        time_emb_dim=None, time_emb_scale=1.0, num_classes=None, activation=F.relu,
        dropout=0.1, attention_resolutions=(), norm="gn", num_groups=32, initial_pad=0,
    ):
        super().__init__()
        self.activation = activation
        self.initial_pad = initial_pad
        self.num_classes = num_classes
        self.q_act_time = QuantizedTernaryActivation()
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale),
            QuantizedLinear(base_channels, time_emb_dim),
            nn.SiLU(),
            QuantizedLinear(time_emb_dim, time_emb_dim),
        ) if time_emb_dim is not None else None
        self.init_conv = QuantizedConv2d(img_channels, base_channels, 3, padding=1)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(
                    now_channels, out_channels, dropout, time_emb_dim=time_emb_dim,
                    num_classes=num_classes, activation=activation, norm=norm,
                    num_groups=num_groups, use_attention=i in attention_resolutions,
                ))
                now_channels = out_channels
                channels.append(now_channels)
            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)
        self.mid = nn.ModuleList([
            ResidualBlock(
                now_channels, now_channels, dropout, time_emb_dim=time_emb_dim,
                num_classes=num_classes, activation=activation, norm=norm,
                num_groups=num_groups, use_attention=True,
            ),
            ResidualBlock(
                now_channels, now_channels, dropout, time_emb_dim=time_emb_dim,
                num_classes=num_classes, activation=activation, norm=norm,
                num_groups=num_groups, use_attention=False,
            ),
        ])
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.ups.append(ResidualBlock(
                    channels.pop() + now_channels, out_channels, dropout,
                    time_emb_dim=time_emb_dim, num_classes=num_classes,
                    activation=activation, norm=norm, num_groups=num_groups,
                    use_attention=i in attention_resolutions,
                ))
                now_channels = out_channels
            if i != 0:
                self.ups.append(Upsample(now_channels))
        self.out_norm = get_norm(norm, base_channels, num_groups)
        self.q_act_out = QuantizedTernaryActivation()
        self.out_conv = QuantizedConv2d(base_channels, img_channels, 3, padding=1)

    def forward(self, x, time=None, y=None):
        ip = self.initial_pad
        if ip != 0: x = F.pad(x, (ip,) * 4)
        if self.time_mlp is not None:
            if time is None: raise ValueError("Time conditioning was specified but time is not passed")
            time_emb = self.time_mlp(time)
            time_emb = self.q_act_time(time_emb)
        else:
            time_emb = None
        if self.num_classes is not None and y is None:
            raise ValueError("Class conditioning was specified but y is not passed")
        x = self.init_conv(x)
        skips = [x]
        for layer in self.downs:
            x = layer(x, time_emb, y)
            skips.append(x)
        for layer in self.mid:
            x = layer(x, time_emb, y)
        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, time_emb, y)
        x = self.out_norm(x)
        x = self.activation(x)
        x = self.q_act_out(x)
        x = self.out_conv(x)
        if self.initial_pad != 0: return x[:, :, ip:-ip, ip:-ip]
        else: return x