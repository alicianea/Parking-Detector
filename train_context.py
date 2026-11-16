"""Train or evaluate slot context using CNR-EXT full frames and patch labels."""
import argparse
import csv
from pathlib import Path
import random

import cv2
import torch
from torch.nn import functional as F

import main as detector
from slot_context import SlotContext, checkpoint_digest, load_context, slot_geometry

# Group the patch labels by full camera frame so each frame can form one graph.
def read_frames(label_path):
    frames = {}
    with label_path.open() as file:
        for line in file:
            if not line.strip():
                continue
            patch, label = line.rsplit(maxsplit=1)
            patch = Path(patch)
            # The patch filename identifies the capture time and parking slot.
            # Keep its weather, date, and camera folders when forming the full-image path.
            _, date, time, _, slot = patch.stem.split("_")
            frame = patch.with_name(f"{date}_{time.replace('.', '')}.jpg")
            slot, label = int(slot), int(label)
            # Each frame maps slot IDs to labels: 0 is free, 1 is occupied.
            labels = frames.setdefault(frame, {})
            if label not in (0, 1) or (slot in labels and labels[slot] != label):
                raise ValueError(f"Invalid or conflicting occupancy label for {patch}")
            labels[slot] = label
    if not frames:
        raise ValueError(f"No frames in {label_path}")
    return frames

# Keep all slots from one frame in the same split to avoid sharing that frame
# between training, validation, and testing.
def verify_splits(splits):
    seen = set()
    for split in splits:
        overlap = seen.intersection(split)
        if overlap:
            raise ValueError(f"Label splits share {len(overlap)} frames, e.g. {min(overlap)}. "
                             "Keep every slot from a frame in the same split.")
        seen.update(split)


def full_frame_path(image_root, relative_path):
    image_path = image_root / relative_path
    if image_path.is_file():
        return image_path
    # Also accept filenames that use HH.MM instead of HHMM.
    stem = image_path.stem
    dotted = image_path.with_name(f"{stem[:-2]}.{stem[-2:]}.jpg")
    if dotted.is_file():
        return dotted
    raise FileNotFoundError(f"Missing full frame: {image_path} (also tried {dotted.name})")


def check_inputs(frames, image_root, csv_root):
    # Check inputs before encoding. Read the slot IDs from each camera CSV once.
    camera_slots = {}
    for relative_path, labels in frames.items():
        full_frame_path(image_root, relative_path)
        camera = relative_path.parent.name
        if camera not in camera_slots:
            with (csv_root / (camera + ".csv")).open(newline="") as file:
                slots = [int(row["SlotId"]) for row in csv.DictReader(file)]
            if len(set(slots)) != len(slots):
                raise ValueError(f"Duplicate slot IDs in {camera}.csv")
            camera_slots[camera] = set(slots)
        missing = labels.keys() - camera_slots[camera]
        if missing:
            raise ValueError(f"Missing slot IDs {sorted(missing)} in {camera}.csv for {relative_path}")
    print(f"Matched paths and slot IDs for {len(frames)} frames", flush=True)


def encode_frames(frames, image_root, csv_root, baseline):
    encoded = []
    for index, (relative_path, labels) in enumerate(sorted(frames.items()), 1):
        image_path = full_frame_path(image_root, relative_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load matching full frame: {image_path}")
        csv_path = csv_root / (relative_path.parent.name + ".csv")
        # Reuse the inference crop pipeline so training sees the same kind of inputs.
        crops, spaces = detector.extract_frame_slots(image, csv_path)
        slot_ids = [int(space["id"]) for space, *_ in spaces]
        unmatched = labels.keys() - set(slot_ids)
        if unmatched:
            raise ValueError(f"Missing usable boxes for slots {sorted(unmatched)} in {csv_path} "
                             f"for frame {relative_path}")
        # Use crop order for every tensor. Unlabeled slots can send messages,
        # and cross_entropy ignores their -100 targets.
        targets = torch.tensor([labels.get(slot, -100) for slot in slot_ids])
        # Run MobileNetV2 once per frame without gradients and cache its outputs
        # on the CPU. Each epoch reuses them instead of reprocessing the images.
        features, logits = SlotContext.encode(baseline, crops.to(detector.device))
        encoded.append((features.cpu(), logits.cpu(), slot_geometry(spaces, image.shape), targets))
        if index % 100 == 0 or index == len(frames):
            print(f"Encoded {index}/{len(frames)} frames", flush=True)
    return encoded


@torch.no_grad()
def evaluate(context, frames, name):
    context.eval()
     # Track one confusion matrix per model. Rows are true labels and columns
    # are predicted labels, both ordered free then occupied.
    confusion = torch.zeros(2, 2, 2, dtype=torch.long)
    for frame in frames:
        features, baseline, geometry, targets = move_frame(frame)
        predictions = context(features, baseline, geometry)
        # Score both models on the same labeled slots.
        mask = targets >= 0
        for matrix, scores in zip(confusion, (baseline, predictions)):
            # Map each (true, predicted) pair to a bin in TN, FP, FN, TP order.
            pairs = 2 * targets[mask] + scores[mask].argmax(dim=1)
            matrix += torch.bincount(pairs.cpu(), minlength=4).reshape(2, 2)
    for method, matrix in zip(("MobileNetV2", name), confusion):
        tn, fp, fn, tp = matrix.flatten().tolist()
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        print(f"{method}: accuracy={accuracy:.2%}, occupied F1={f1:.2%}, "
              f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    # The diagonal counts correct context predictions. Every epoch uses the same
    # validation labels, so this count ranks epochs the same way as accuracy.
    return confusion[1].trace().item()


def move_frame(frame):
    # Move one frame from the CPU cache to the model device for training or evaluation.
    return tuple(tensor.to(detector.device) for tensor in frame)


def run(args):
    # Seed model initialization and the separate random generator for shuffling.
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train = read_frames(args.labels / "train.txt")
    val = read_frames(args.labels / "val.txt")
    test_path = args.labels / "test.txt"
    test = read_frames(test_path) if args.evaluate or test_path.exists() else {}
    verify_splits((train, val, test))
    # The data check finishes before loading any model weights.
    if args.check_data:
        check_inputs(train | val | test, args.images, args.camera_csvs)
        return
    # A subset can restrict scoring to selected test slots, such as occluded ones.
    # Encoding still uses all valid crops in each selected frame.
    if args.subset:
        subset = read_frames(args.subset)
        for frame, labels in subset.items():
            if labels.items() - test.get(frame, {}).items():
                raise ValueError(f"Subset labels must also occur in test.txt: {frame}")
        test = subset
    check_inputs(test if args.evaluate else train | val, args.images, args.camera_csvs)
    # The detector loads the existing MobileNetV2 checkpoint in evaluation mode.
    baseline, mapping = detector.load_model()
    if mapping != {"free": 0, "busy": 1}:
        raise ValueError("Baseline class mapping must match CNR labels (free=0, busy=1).")
    # Test mode loads a saved refinement model and skips training.
    if args.evaluate:
        context = load_context(args.checkpoint, detector.MODEL_PATH, detector.device)
        frames = encode_frames(test, args.images, args.camera_csvs, baseline)
        evaluate(context, frames, "Geometry + GNN" if context.neighbors else "Geometry only")
        return
    # Only the context head is passed to the optimizer and updated during training.
    train_frames = encode_frames(train, args.images, args.camera_csvs, baseline)
    val_frames = encode_frames(val, args.images, args.camera_csvs, baseline)
    context = SlotContext(baseline.last_channel, args.neighbors).to(detector.device)
    optimizer = torch.optim.Adam(context.parameters(), lr=0.001)
    baseline_sha256 = checkpoint_digest(detector.MODEL_PATH)
    best = -1
    for epoch in range(args.epochs):
        context.train()
        # Shuffle whole frames so every group of parking slots stays together.
        rng.shuffle(train_frames)
        for frame in train_frames:
            features, logits, geometry, targets = move_frame(frame)
            # Clear the previous frame’s gradients, compute the labeled-slot loss,
            # and update the context head once for this frame.
            optimizer.zero_grad()
            loss = F.cross_entropy(context(features, logits, geometry), targets)
            loss.backward()
            optimizer.step()
        print(f"Validation after epoch {epoch + 1}/{args.epochs}")
        score = evaluate(context, val_frames, "Geometry + GNN" if args.neighbors else "Geometry only")
        # Save only when validation improves. The baseline hash lets inference
        # verify that it is using the exact MobileNetV2 checkpoint from training.
        if score > best:
            best = score
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "context_state_dict": context.state_dict(),
                "feature_size": baseline.last_channel,
                "neighbors": args.neighbors,
                "baseline_sha256": baseline_sha256,
            }, args.checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=detector.PROJECT_DIR /
                        "dataset/CNR-EXT-Patches-150x150/LABELS")
    parser.add_argument("--images", type=Path, default=detector.DATA_DIR)
    parser.add_argument("--camera-csvs", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=detector.PROJECT_DIR /
                        "models/slot_context.pth")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate checkpoint on test.txt")
    parser.add_argument("--check-data", action="store_true", help="Check frame paths and slot IDs")
    parser.add_argument("--subset", type=Path, help="Evaluate a subset of test.txt, e.g. occluded slots")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=4,
                        help="Use 0 to train the geometry-only comparison")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # In the default dataset layout, camera CSVs sit beside the images folder.
    args.camera_csvs = args.camera_csvs or args.images.parent
    if args.epochs < 1 or args.neighbors < 0:
        parser.error("--epochs must be positive and --neighbors must be nonnegative")
    if args.subset and not args.evaluate:
        parser.error("--subset requires --evaluate")
    run(args)
