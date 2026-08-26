from torchvision.models import resnet50, ResNet50_Weights

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, StepLR
from torch.optim import AdamW

from torch import nn
import torch

from pathlib import Path

WARMUP_EPOCHS = 2
MAIN_EPOCHS = 5

model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 4)

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

warmup = LinearLR(
    optimizer,
    start_factor=0.01,
    end_factor=1.0,
    total_iters=WARMUP_EPOCHS,
)

cosine = CosineAnnealingLR(
    optimizer,
    T_max=MAIN_EPOCHS - WARMUP_EPOCHS,
    eta_min=1e-5,
)

scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup, cosine],
    milestones=[WARMUP_EPOCHS],
)

criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

path = Path("../../results/Model.pth")

if __name__ == "__main__":
    ...