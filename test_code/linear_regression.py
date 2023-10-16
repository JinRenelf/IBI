import torch
import torch.nn as nn
import torch.optim as optim


# Define model
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


if __name__ == '__main__':
    # Data
    x_data = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    y_data = torch.tensor([[2.0], [4.0], [6.0]], dtype=torch.float32)

    # Model parameters
    input_size = 1
    output_size = 1
    learning_rate = 0.01
    num_epochs = 100

    # Move data to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_data = x_data.to(device)
    y_data = y_data.to(device)

    # Create model and move it to multiple GPUs (if available)
    model = LinearModel()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(x_data)
        loss = criterion(outputs, y_data)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Inference
    x_new = torch.tensor([[4.0]], dtype=torch.float32).to(device)
    predicted = model(x_new)
    print(f'Predicted value: {predicted.item()}')
