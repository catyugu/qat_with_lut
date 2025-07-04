import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the necessary QAT components from QAT_UNet.py
from .QAT_UNet import ScaledWeightTernary, ScaledActivationTernary, QATConv2d

# Custom QuantizedLinear to align with QATConv2d's weight quantization
class QATLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantize_weights = True # Enable/disable quantization for this layer

    def forward(self, x):
        if self.quantize_weights and self.training:
            # Apply ScaledWeightTernary to weights during training
            quantized_weight = ScaledWeightTernary.apply(self.weight)
            return F.linear(x, quantized_weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)

# The QAT Classifier model
class QAT_Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(QAT_Classifier, self).__init__()
        
        # Using QATConv2d and QATLinear defined in this context
        self.conv1 = QATConv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Removed instantiation of ScaledActivationTernary as it's an autograd.Function
        
        self.conv2 = QATConv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Removed instantiation of ScaledActivationTernary
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv3 = QATConv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # Removed instantiation of ScaledActivationTernary
        
        self.conv4 = QATConv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        # Removed instantiation of ScaledActivationTernary
        
        self.fc1 = QATLinear(128 * 8 * 8, 512) # Using our custom QATLinear
        self.bn_fc1 = nn.BatchNorm1d(512)
        # Removed instantiation of ScaledActivationTernary
        
        self.fc2 = QATLinear(512, num_classes) # Using our custom QATLinear
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        # Apply ScaledActivationTernary directly using .apply()
        x = F.relu(ScaledActivationTernary.apply(x))
        
        x = self.conv2(x)
        x = self.bn2(x)
        # Apply ScaledActivationTernary directly using .apply()
        x = F.relu(ScaledActivationTernary.apply(x))
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        # Apply ScaledActivationTernary directly using .apply()
        x = F.relu(ScaledActivationTernary.apply(x))
        
        x = self.conv4(x)
        x = self.bn4(x)
        # Apply ScaledActivationTernary directly using .apply()
        x = F.relu(ScaledActivationTernary.apply(x))
        x = self.pool(x)
        
        x = x.view(-1, 128 * 8 * 8)
        
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        # Apply ScaledActivationTernary directly using .apply()
        x = F.relu(ScaledActivationTernary.apply(x))
        
        x = self.fc2(x)
        return x

    def set_quantization_enabled(self, enabled: bool):
        """Toggles quantization for all QATConv2d and QATLinear layers."""
        for module in self.modules():
            if isinstance(module, (QATConv2d, QATLinear)):
                module.quantize_weights = enabled
