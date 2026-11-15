
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from pathlib import Path

from parking_dataset import ParkingDataset

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "dataset" / "CNR-EXT-Patches-150x150"
MODEL_PATH = PROJECT_DIR / "models" / "parking_classifier.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Evaluation device:", device)

# Match the preprocessing used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load held-out test data
test_dataset = ParkingDataset(
    data_dir=DATA_DIR,
    label_file="test.txt",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# Recreate the same architecture used in training
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(
    model.last_channel, 2
)

# Load trained weights
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# Confusion matrix:
# Rows = actual labels, columns = predicted labels
confusion = torch.zeros(2, 2, dtype=torch.long)

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu()

        for actual, predicted in zip(labels, predictions):
            confusion[actual, predicted] += 1

tn = confusion[0, 0].item()
fp = confusion[0, 1].item()
fn = confusion[1, 0].item()
tp = confusion[1, 1].item()

def safe_divide(a, b):
    return a / b if b else 0.0

accuracy = safe_divide(tp + tn, confusion.sum().item())
precision = safe_divide(tp, tp + fp)
recall = safe_divide(tp, tp + fn)
f1 = safe_divide(2 * precision * recall, precision + recall)

print("\nTest Results")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")

print("\nConfusion Matrix")
print(confusion)
