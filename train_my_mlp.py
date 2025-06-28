import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os

class TernaryQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, threshold=0.1):
        # Quantize input_tensor to ternary values (-1, 0, 1)
        # Values above threshold become 1.0, below -threshold become -1.0, otherwise 0.0
        return torch.where(input_tensor > threshold, 1.0, torch.where(input_tensor < -threshold, -1.0, 0.0))

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator (STE): The gradient is passed directly through
        # the non-differentiable quantization operation.
        return grad_output, None # None for the threshold argument

class TernaryLinear(nn.Linear):
    def forward(self, input):
        # Apply ternary quantization to the weights during the forward pass.
        # The stored self.weight remains float for gradient updates.
        quantized_weight = TernaryQuantizeSTE.apply(self.weight)
        return F.linear(input, quantized_weight, self.bias)

class ActivationQuantizer(nn.Module):
    def __init__(self):
        super(ActivationQuantizer, self).__init__()
        # Register a buffer to store the activation scale.
        # A buffer is part of the state_dict but not a trainable parameter.
        self.register_buffer('scale', torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x):
        # Only update the scale during training
        if self.training:
            # Calculate the dynamic scale based on the absolute maximum value of the activations.
            # This maps the activation range to [-127, 127] for int8.
            max_abs_val = x.abs().max()
            # Avoid division by zero or very small numbers
            if max_abs_val < 1e-6:
                self.scale.data[()] = 1.0
            else:
                self.scale.data[()] = 127.0 / max_abs_val

        # Quantize the activations to int8 equivalent and then dequantize them back to float.
        # This simulates the quantization effects during the forward pass.
        x_quant = (x * self.scale).round().clamp(-128, 127) # Quantize to int8 range
        x_dequant = x_quant / self.scale # Dequantize back to float

        # Straight-Through Estimator for activations:
        # The identity function `x` is used for the forward pass for gradient calculation,
        # but the dequantized value `x_dequant` is used for the subsequent computation,
        # while detaching it to prevent gradients from flowing through the rounding/clamping.
        return x + (x_dequant - x).detach()

class SmallMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super(SmallMLP, self).__init__()
        self.input_dim = input_dim
        self.fc1 = TernaryLinear(input_dim, hidden_dim)
        # Activation quantizer is applied after ReLU for the output of the first layer
        self.act_quant1 = ActivationQuantizer()
        self.fc2 = TernaryLinear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim) # Flatten input images
        x = self.fc1(x)           # First linear layer with ternary weights
        x = F.relu(x)             # ReLU activation
        x = self.act_quant1(x)    # Activation quantization and dequantization
        x = self.fc2(x)           # Second linear layer with ternary weights
        return F.log_softmax(x, dim=1) # LogSoftmax for classification output

def train_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Data path setup: Assuming FashionMNIST data will be downloaded to a 'datasets/F_MNIST_data' folder
    data_path = os.path.join(script_dir, 'datasets', 'F_MNIST_data')
    os.makedirs(data_path, exist_ok=True) # Create directory if it doesn't exist

    # Define the transformation for the FashionMNIST images
    # ToTensor() converts PIL Image to FloatTensor and scales to [0.0, 1.0]
    # Normalize() scales to [-1.0, 1.0] using mean 0.5 and std dev 0.5
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load training dataset
    trainset = datasets.FashionMNIST(data_path, download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Load test dataset for evaluation during training
    testset = datasets.FashionMNIST(data_path, download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    model = SmallMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss() # Suitable for log_softmax output combined with labels

    epochs = 5
    print("Starting training with Weight AND Activation Quantization (QAT)...")
    for e in range(epochs):
        model.train() # Set model to training mode
        running_loss = 0; correct_train_preds = 0; total_train_preds = 0
        for images, labels in trainloader:
            optimizer.zero_grad() # Clear gradients
            output = model(images) # Forward pass
            loss = criterion(output, labels) # Calculate loss
            loss.backward() # Backward pass (compute gradients)
            optimizer.step() # Update weights

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1) # Get predicted class
            total_train_preds += labels.size(0)
            correct_train_preds += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train_preds / total_train_preds
        print(f"Epoch {e+1}/{epochs}.. Train Loss: {running_loss/len(trainloader):.4f}, Train Acc: {train_accuracy:.2f}%")

        # Evaluate model on the test set after each epoch
        model.eval() # Set model to evaluation mode (disables dropout, batch norm updates, etc.)
        test_correct_preds = 0; test_total_preds = 0
        with torch.no_grad(): # Disable gradient calculations for evaluation
            for images, labels in testloader:
                output = model(images)
                _, predicted = torch.max(output.data, 1)
                test_total_preds += labels.size(0)
                test_correct_preds += (predicted == labels).sum().item()

        test_accuracy = 100 * test_correct_preds / test_total_preds
        # This test_accuracy corresponds to the "Standard Float MLP" (dequantized float behavior) in C++
        print(f"Epoch {e+1}/{epochs}.. Test Acc (QAT Float): {test_accuracy:.2f}%")

    print("\nTraining finished.")
    # Save the state_dict (weights and activation_quantizer scale)
    torch.save(model.state_dict(), 'small_mlp_full_qat.pth')
    print("Model saved to 'small_mlp_full_qat.pth'")

if __name__ == '__main__':
    train_model()
