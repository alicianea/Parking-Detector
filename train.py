"""Train a MobileNetV2 parking occupancy baseline."""
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms

from parking_dataset import ParkingDataset


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "dataset" / "CNR-EXT-Patches-150x150"
CLASS_TO_IDX = {"free": 0, "busy": 1}


@torch.no_grad()
def cache_features(backbone, loader, device, split, feature_size):
    # Allocate once on the CPU, then fill one batch at a time.
    features = torch.empty(len(loader.dataset), feature_size)
    targets = torch.empty(len(loader.dataset), dtype=torch.long)
    start = 0
    for batch, (images, labels) in enumerate(loader, 1):
        vectors = backbone(images.to(device, non_blocking=True)).mean(dim=(2, 3))
        end = start + labels.size(0)
        features[start:end].copy_(vectors.cpu())
        targets[start:end].copy_(labels)
        start = end
        if batch % 100 == 0 or start == len(features):
            print(f"Cached {start}/{len(features)} {split} crops", flush=True)
    return TensorDataset(features, targets)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}", flush=True)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    resize = transforms.Resize((224, 224))
    eval_transform = transforms.Compose([resize, transforms.ToTensor(), normalize])
    train_transform = eval_transform
    if args.fine_tune_blocks:
        train_transform = transforms.Compose([
            resize,
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            normalize,
        ])
    train_set = ParkingDataset(DATA_DIR, "train.txt", transform=train_transform)
    val_set = ParkingDataset(DATA_DIR, "val.txt", transform=eval_transform)
    if len(train_set) == 0 or len(val_set) == 0:
        raise ValueError("Training and validation sets must both contain images.")

    options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0 and args.fine_tune_blocks > 0,
    }
    train_loader = DataLoader(train_set, shuffle=bool(args.fine_tune_blocks), **options)
    val_loader = DataLoader(val_set, **options)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    if args.fine_tune_blocks > len(model.features):
        raise ValueError(f"MobileNetV2 has only {len(model.features)} feature blocks.")
    model.features.requires_grad_(False)
    if args.fine_tune_blocks:
        model.features[-args.fine_tune_blocks:].requires_grad_(True)
    model.to(device)
    model.features.eval()

    predictor = model
    groups = [{"params": model.classifier.parameters(), "lr": 1e-3}]
    if args.fine_tune_blocks:
        groups.append({"params": model.features[-args.fine_tune_blocks:].parameters(), "lr": 1e-4})
    else:
        # The frozen encoder and plain transforms yield the same features each epoch.
        train_vectors = cache_features(model.features, train_loader, device, "training", model.last_channel)
        val_vectors = cache_features(model.features, val_loader, device, "validation", model.last_channel)
        train_loader = DataLoader(train_vectors, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_vectors, batch_size=args.batch_size)
        predictor = model.classifier

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0

    for epoch in range(args.epochs):
        predictor.train()
        if args.fine_tune_blocks:
            # Keep the BatchNorm statistics in the frozen prefix fixed.
            model.features[:-args.fine_tune_blocks].eval()
        running_loss = torch.zeros((), device=device)
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(predictor(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach() * labels.size(0)
        scheduler.step()

        predictor.eval()
        correct = torch.zeros((), dtype=torch.long, device=device)
        with torch.inference_mode():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                correct += (predictor(inputs).argmax(dim=1) == labels).sum()

        val_accuracy = 100 * correct.item() / len(val_set)
        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Loss: {running_loss.item() / len(train_set):.4f} | "
              f"Val Accuracy: {val_accuracy:.2f}%", flush=True)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": CLASS_TO_IDX,
            }, args.output)
            print(f"Saved {args.output}", flush=True)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fine-tune-blocks", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "models" / "parking_classifier_fast.pth",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.workers < 0 or args.fine_tune_blocks < 0:
        parser.error("--workers and --fine-tune-blocks must be nonnegative")

    train(args)
