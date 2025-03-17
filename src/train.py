import torch # type: ignore
import torch.nn as nn
import torch.optim as optim # type: ignore
from torchvision import datasets, transforms

# Define a simple CNN model
class DentalNet(nn.Module):
    def __init__(self):
        super(DentalNet, self).__init__()
        self.fc1 = nn.Linear(224 * 224, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes (Cavity / No Cavity)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model
model = DentalNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("✅ PyTorch model ready!")
