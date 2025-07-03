import torch.nn as nn
import torch.nn.functional as F
from .QAT_UNet import QuantizedConv2d, QuantizedLinear, QuantizedTernaryActivation

class QAT_Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(QAT_Classifier, self).__init__()
        
        # --- Layers with BatchNorm for stability ---
        self.conv1 = QuantizedConv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.q_act1 = QuantizedTernaryActivation()
        
        self.conv2 = QuantizedConv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.q_act2 = QuantizedTernaryActivation()
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv3 = QuantizedConv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.q_act3 = QuantizedTernaryActivation()
        
        self.conv4 = QuantizedConv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.q_act4 = QuantizedTernaryActivation()
        
        self.fc1 = QuantizedLinear(128 * 8 * 8, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.q_act5 = QuantizedTernaryActivation()
        
        self.fc2 = QuantizedLinear(512, num_classes)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # --- Forward Pass with BatchNorm -> Quantize -> ReLU order ---
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(self.q_act1(x))
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(self.q_act2(x))
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(self.q_act3(x))
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(self.q_act4(x))
        x = self.pool(x)
        
        # Flatten feature map for the fully-connected layers
        x = x.view(-1, 128 * 8 * 8)
        
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(self.q_act5(x))
        
        x = self.fc2(x)
        return x
