
import torch
import torch.nn as nn
from torchvision import models, transforms
from parking_dataset import ParkingDataset
from torch.utils.data import DataLoader
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "dataset" / "CNR-EXT-Patches-150x150"

PATCHES_DIR = DATA_DIR / "PATCHES"
LABELS_DIR = DATA_DIR / "LABELS"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ImageNet preprocessing for pretrained MobileNetV2
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = ParkingDataset(
    data_dir=DATA_DIR,
    label_file="train.txt",
    transform=transform
)

val_dataset = ParkingDataset(
    data_dir=DATA_DIR,
    label_file="val.txt",
    transform=transform
)

class_to_idx = {
    "free": 0,
    "busy": 1
}

print("Class mapping:", class_to_idx)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True
)

val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False
)

# Load pretrained MobileNetV2
weights = models.MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)

# Freeze pretrained feature extractor
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier with binary classification head
model.classifier[1] = nn.Linear(
    model.last_channel, 2
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)


epochs = 2

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    train_loss = running_loss / len(train_dataset)
    val_accuracy = 100 * correct / total

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {train_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

torch.save({
    "model_state_dict": model.state_dict(),
    "class_to_idx": class_to_idx
}, MODEL_DIR / "parking_classifier.pth")