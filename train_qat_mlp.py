import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os

# Define the ternary quantization threshold for weights.
TERNARY_THRESHOLD = 0.001

class TernaryQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        # This STE is now only used for weights, which are already clamped to [-1, 1]
        return torch.where(input_tensor > TERNARY_THRESHOLD, 1.0,
                           torch.where(input_tensor < -TERNARY_THRESHOLD, -1.0, 0.0))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class TernaryLinear(nn.Linear):
    def forward(self, input):
        clipped_weight = self.weight.clamp(-1.0, 1.0)
        quantized_weight = TernaryQuantizeSTE.apply(clipped_weight)
        return F.linear(input, quantized_weight, self.bias)

# --- NEW: QAT Activation Module with Learned Scale ---
class FakeQuantTernary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        # Quantize to int8 range, then ternarize
        x_quant = torch.clamp(torch.round(x / scale), -128, 127)

        # Ternarize based on a small threshold in the quantized space
        # This is equivalent to `convert_int8_to_ternary_activation` in C++
        x_ternary = torch.where(x_quant > 1, 1.0, torch.where(x_quant < -1, -1.0, 0.0))

        # Dequantize to simulate the quantization error during forward pass
        x_dequant = x_ternary * scale
        return x_dequant

    @staticmethod
    def backward(ctx, grad_output):
        # Pass gradients straight through. None for the 'scale' input.
        return grad_output, None

class QuantizedTernaryActivation(nn.Module):
    def __init__(self, momentum=0.1):
        super().__init__()
        self.momentum = momentum
        # This buffer will store the learned maximum absolute value of the activation
        self.register_buffer('running_abs_max', torch.tensor(1.0))

    def forward(self, x):
        if self.training:
            # Update the running max with the current batch's max value
            current_max = torch.max(torch.abs(x.detach()))
            self.running_abs_max.mul_(1.0 - self.momentum).add_(self.momentum * current_max)

        # The scale is the range [-max, max] divided by the bits of precision (2*127)
        scale = self.running_abs_max / 127.0
        return FakeQuantTernary.apply(x, scale)

class SmallMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=320, output_dim=10):
        super(SmallMLP, self).__init__()
        self.input_dim = input_dim
        self.fc1 = TernaryLinear(input_dim, hidden_dim)
        # Use the new QAT activation module
        self.activation = QuantizedTernaryActivation()
        self.fc2 = TernaryLinear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.fc1(x)
        x = self.activation(x) # Apply QAT activation
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'datasets', 'F_MNIST_data')
    os.makedirs(data_path, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.FashionMNIST(data_path, download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = datasets.FashionMNIST(data_path, download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    model = SmallMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.NLLLoss()

    epochs = 3 # Increased epochs for QAT to stabilize
    print(f"Starting QAT training with learned activation scale...")
    for e in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        test_correct_preds = 0; total_train_preds = 0
        with torch.no_grad():
            for images, labels in testloader:
                output = model(images)
                _, predicted = torch.max(output.data, 1)
                total_train_preds += labels.size(0)
                test_correct_preds += (predicted == labels).sum().item()

        test_accuracy = 100 * test_correct_preds / total_train_preds
        print(f"Epoch {e+1}/{epochs}.. Loss: {running_loss/len(trainloader):.4f}, Test Acc: {test_accuracy:.2f}%")
        # Print the learned scale for observation
        print(f"  Learned activation abs_max: {model.activation.running_abs_max.item():.4f}")

    print("\nTraining complete.")
    torch.save(model.state_dict(), 'small_mlp_ternary_act_qat.pth')
    print("Model saved to 'small_mlp_ternary_act_qat.pth'")

if __name__ == '__main__':
    train_model()
