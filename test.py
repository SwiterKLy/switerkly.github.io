import torch
import torch.nn as nn
import torch.nn.functional as F

# Визначення кастомних класів
class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 50 * 50, 256),  # Можливо треба буде підкорегувати
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DistanceLayer(nn.Module):
    def forward(self, x1, x2):
        return F.pairwise_distance(x1, x2)

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_net=None):
        super().__init__()
        self.embedding_net = embedding_net or EmbeddingNet()
        self.distance = DistanceLayer()

    def forward(self, x1, x2):
        embed1 = self.embedding_net(x1)
        embed2 = self.embedding_net(x2)
        return embed1, embed2

# ✅ Додаємо клас в whitelist
torch.serialization.add_safe_globals({
    'SiameseNetwork': SiameseNetwork,
    'EmbeddingNet': EmbeddingNet,
    'DistanceLayer': DistanceLayer
})

# Визначення пристрою
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Завантаження
model = torch.load('Model.pth', map_location=device, weights_only=False)
print("Модель завантажена:", type(model))
