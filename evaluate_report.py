"""Evaluate the three bundled models on the same held-out CNR-EXT slots."""
import argparse
import csv
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

import main as detector
from slot_context import SlotContext, checkpoint_digest, load_context, nearest_slots, slot_geometry
from train_context import check_inputs, full_frame_path, read_frames, verify_splits

WEATHERS = ("SUNNY", "OVERCAST", "RAINY")
MODELS = ("MobileNetV2", "Geometry-only", "GNN")
CASES = ("baseline_fixed", "geometry_fixed", "new_vs_baseline", "new_vs_geometry")
CLASSES = ("Vacant", "Occupied")
MODEL_DIR = detector.PROJECT_DIR / "models"


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save {path}")


def center(space):
    return round(space["x"] + space["w"] / 2), round(space["y"] + space["h"] / 2)


def outline(image, slot, color):
    space, *box = slot
    detector.draw_slot_outline(image, box, space.get("polygon"), color, 3)


def graph_figure(image, slots, target, neighbor_indices):
    annotated = image.copy()
    source = center(slots[target][0])
    for index in neighbor_indices:
        destination = center(slots[index][0])
        cv2.arrowedLine(annotated, destination, source, (255, 220, 30), 3,
                        cv2.LINE_AA, tipLength=0.16)
        outline(annotated, slots[index], (255, 220, 30))
        cv2.putText(annotated, str(slots[index][0]["id"]), destination,
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 220, 30), 2, cv2.LINE_AA)
    outline(annotated, slots[target], (0, 220, 255))
    banner = cv2.copyMakeBorder(annotated, 56, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    cv2.putText(banner, f"Incoming messages to slot {slots[target][0]['id']}",
                (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(banner, "Cyan = selected neighbors   |   Yellow = receiving slot",
                (12, 46), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
    return banner


def case_figure(image, slot, row):
    highlighted = image.copy()
    outline(highlighted, slot, (0, 220, 255))
    top = cv2.copyMakeBorder(highlighted, 115, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    lines = (
        f"{row['frame']}  |  Slot {row['slot']}",
        f"Ground truth: {CLASSES[row['truth']]}",
        f"MobileNetV2: {CLASSES[row['MobileNetV2']]}   |   "
        f"Geometry-only: {CLASSES[row['Geometry-only']]}   |   GNN: {CLASSES[row['GNN']]}",
    )
    for y, line in zip((27, 58, 89), lines):
        cv2.putText(top, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, .62,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return top


def comparison_cases(truth, baseline, geometry, gnn):
    return (
        baseline != truth and gnn == truth,
        geometry != truth and gnn == truth,
        gnn != truth and baseline == truth,
        gnn != truth and geometry == truth,
    )


@torch.inference_mode()
def run(args):
    label_paths = [args.labels / f"{split}.txt" for split in ("train", "val", "test")]
    required = [*label_paths, args.images, args.camera_csvs]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing evaluation inputs:\n  " + "\n  ".join(missing))
    train, val, test = map(read_frames, label_paths)
    verify_splits((train, val, test))
    counts = dict.fromkeys(("FULL", *WEATHERS), 0)
    for frame in test:
        weather = frame.parts[0]
        if weather not in WEATHERS or frame.is_absolute() or ".." in frame.parts:
            raise ValueError(f"Unexpected test frame path: {frame}")
        counts[weather] += 1
        counts["FULL"] += 1
    for weather in WEATHERS:
        if not counts[weather]:
            raise ValueError(f"Test split has no {weather} frames")
    check_inputs(test, args.images, args.camera_csvs)

    baseline, mapping = detector.load_model()
    if mapping != {"free": 0, "busy": 1}:
        raise ValueError(f"Unexpected baseline class mapping: {mapping}")
    encoder_device = torch.device(args.encoder_device or detector.device)
    baseline = baseline.to(encoder_device)
    head_paths = [MODEL_DIR / "slot_geometry.pth", MODEL_DIR / "slot_context.pth"]
    geometry_head, gnn_head = [
        load_context(path, detector.MODEL_PATH, detector.device) for path in head_paths
    ]
    if (geometry_head.neighbors, gnn_head.neighbors) != (0, 4):
        raise ValueError("Expected the bundled zero-neighbor and four-neighbor heads")

    args.output.mkdir(parents=True, exist_ok=False)
    confusion = defaultdict(lambda: np.zeros((3, 4), dtype=np.int64))
    changes = defaultdict(lambda: dict.fromkeys(CASES, 0))
    examples = []
    saved = set()
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["frame", "slot", "truth", *MODELS],
                                lineterminator="\n")
        writer.writeheader()
        for frame_number, (frame, truth_by_id) in enumerate(sorted(test.items()), 1):
            weather = frame.parts[0]
            image_path = full_frame_path(args.images, frame)
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read {image_path}")
            crops, slots = detector.extract_frame_slots(
                image, args.camera_csvs / f"{frame.parent.name}.csv"
            )
            slot_ids = [int(slot[0]["id"]) for slot in slots]
            if truth_by_id.keys() - set(slot_ids):
                raise ValueError(f"Missing usable labeled slots in {frame}")
            geometry = slot_geometry(slots, image.shape).to(detector.device)
            features, logits = SlotContext.encode(baseline, crops.to(encoder_device))
            features, logits = features.to(detector.device), logits.to(detector.device)
            scores = (logits, geometry_head(features, logits, geometry),
                      gnn_head(features, logits, geometry))
            if any(not torch.isfinite(score).all() for score in scores):
                raise ValueError(f"Non-finite logits in {frame}")
            predictions = torch.stack([score.argmax(dim=1) for score in scores], dim=1)
            for index in sorted(range(len(slots)), key=slot_ids.__getitem__):
                slot_id = slot_ids[index]
                if slot_id not in truth_by_id:
                    continue
                truth = truth_by_id[slot_id]
                labels = predictions[index].tolist()
                row = dict(frame=frame.as_posix(), slot=slot_id, truth=truth)
                row.update(zip(MODELS, labels))
                writer.writerow(row)
                bins = 2 * truth + np.array(labels)
                for group in ("FULL", weather):
                    confusion[group][np.arange(3), bins] += 1
                for category, matched in zip(CASES, comparison_cases(truth, *labels)):
                    if not matched:
                        continue
                    for group in ("FULL", weather):
                        changes[group][category] += 1
                    if category not in saved:
                        saved.add(category)
                        filename = category + ".png"
                        write_image(args.output / filename, case_figure(image, slots[index], row))
                        write_image(args.output / (category + "_original.png"), image)
                        examples.append(dict(category=category, figure=filename, **row))
                        if len(saved) == 1:
                            centers = geometry[:, :2] + geometry[:, 2:] / 2
                            neighbors = nearest_slots(centers, min(gnn_head.neighbors, len(slots) - 1))
                            write_image(args.output / "gnn_neighbors.png",
                                        graph_figure(image, slots, index, neighbors[index].tolist()))
            if frame_number % 100 == 0 or frame_number == len(test):
                print(f"Evaluated {frame_number}/{len(test)} frames", flush=True)

    rows = []
    for group, frames in counts.items():
        for model, (tn, fp, fn, tp) in zip(MODELS, confusion[group].tolist()):
            labeled = tn + fp + fn + tp
            rows.append(dict(split=group, model=model, frames=frames, labeled_slots=labeled,
                             accuracy=(tn + tp) / labeled,
                             occupied_f1=2 * tp / max(1, 2 * tp + fp + fn),
                             TN=tn, FP=fp, FN=fn, TP=tp))
    write_csv(args.output / "metrics.csv", rows)
    write_csv(args.output / "gnn_changes.csv", [dict(split=group, **changes[group]) for group in counts])
    if examples:
        write_csv(args.output / "examples.csv", examples)
    with (args.output / "results.md").open("w", encoding="utf-8") as file:
        print("| Split | Frames | Labeled slots | MobileNetV2 accuracy / F1 | Geometry-only accuracy / F1 | GNN accuracy / F1 |", file=file)
        print("| --- | ---: | ---: | --- | --- | --- |", file=file)
        for i in range(0, len(rows), 3):
            baseline_row = rows[i]
            cells = ["{accuracy:.2%} / {occupied_f1:.2%}".format(**row) for row in rows[i:i + 3]]
            print("| " + " | ".join(map(str, (baseline_row["split"], baseline_row["frames"],
                                             baseline_row["labeled_slots"], *cells))) + " |", file=file)
    manifest = {
        "argv": sys.argv, "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "opencv": cv2.__version__, "numpy": np.__version__, "device": str(detector.device),
        "encoder_device": str(encoder_device),
        "images": str(args.images.resolve()), "camera_csvs": str(args.camera_csvs.resolve()),
        "sha256": {str(path.resolve()): checkpoint_digest(path)
                   for path in [detector.MODEL_PATH, *head_paths, *label_paths,
                                *(args.camera_csvs / f"{camera}.csv"
                                  for camera in sorted({frame.parent.name for frame in test}))]},
        "example_selection": "First matching slot per category in frame-path and numeric slot order.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved evaluation to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=detector.PROJECT_DIR /
                        "dataset/CNR-EXT-Patches-150x150/LABELS")
    parser.add_argument("--images", type=Path, default=detector.DATA_DIR)
    parser.add_argument("--camera-csvs", type=Path)
    parser.add_argument("--encoder-device", choices=("cpu", "cuda", "mps"),
                        help="Override the backbone device, for example mps on Apple Silicon")
    parser.add_argument("--output", type=Path, default=detector.OUTPUT_DIR / "held_out_report")
    args = parser.parse_args()
    args.camera_csvs = args.camera_csvs or args.images.parent
    try:
        run(args)
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
