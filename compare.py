"""Compare MobileNetV2 and GNN parking predictions on one camera frame."""
import argparse
from pathlib import Path

import cv2
import torch

import main as detector
from slot_context import load_context, slot_geometry
from train_context import read_frames


DEFAULT_LABELS = (detector.PROJECT_DIR / "dataset" / "CNR-EXT-Patches-150x150"
                  / "LABELS" / "test.txt")
DEFAULT_CONTEXT = detector.PROJECT_DIR / "models" / "slot_context.pth"
DEFAULT_OUTPUT = detector.PROJECT_DIR / "output" / "parking_comparison.jpg"
WHITE = (255, 255, 255)
BOX_COLORS = {"Occupied": (0, 0, 255), "Vacant": (0, 255, 0)}


def frame_labels(image_path, image_root, label_file):
    # Match the full relative path, which identifies weather, date, and camera.
    try:
        relative = image_path.resolve().relative_to(image_root.resolve())
    except ValueError:
        return {}
    key = relative.with_name(relative.stem.replace(".", "") + ".jpg")
    return read_frames(label_file).get(key, {}) if label_file.is_file() else {}


@torch.inference_mode()
def predict(image, camera_csv, checkpoint):
    baseline, mapping = detector.load_model()
    context = load_context(checkpoint, detector.MODEL_PATH, detector.device)
    crops, spaces = detector.extract_frame_slots(image, camera_csv)
    features, original = context.encode(baseline, crops.to(detector.device))
    geometry = slot_geometry(spaces, image.shape).to(detector.device)
    refined = context(features, original, geometry)
    return spaces, original.argmax(dim=1).tolist(), refined.argmax(dim=1).tolist(), mapping


def draw_panel(image, title, slots, predictions, disagreements, truth, class_names):
    # Put the title above the photo to keep it clear of the parking spaces.
    panel = cv2.copyMakeBorder(image, 64, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    pairs = [(label, truth[int(space["id"])])
             for (space, *_), label in zip(slots, predictions) if int(space["id"]) in truth]
    summary = f"Changed slots: {sum(disagreements)}"
    if pairs:
        summary += f"  |  Correct: {sum(actual == predicted for predicted, actual in pairs)}/{len(pairs)}"
    cv2.putText(panel, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2, cv2.LINE_AA)
    cv2.putText(panel, summary, (12, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.53, WHITE, 1, cv2.LINE_AA)

    for (space, x1, y1, x2, y2), prediction, changed in zip(slots, predictions, disagreements):
        if not changed:
            continue
        status = class_names[prediction]
        cv2.rectangle(panel, (x1, y1 + 64), (x2, y2 + 64), BOX_COLORS[status], 2)
        label = f'{space["id"]}: {status}'
        actual = truth.get(int(space["id"]))
        if actual is not None:
            label += f" (true: {class_names[actual]})"
        (text_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        origin = (max(0, min(x1, panel.shape[1] - text_width - 4)), max(78, y1 + 60))
        cv2.putText(panel, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0),
                    3, cv2.LINE_AA)
        cv2.putText(panel, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE,
                    1, cv2.LINE_AA)
    return panel


def run(args):
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"Could not read image: {args.image}")

    slots, baseline, gnn, mapping = predict(image, args.camera_csv, args.context)
    label_file = args.labels or DEFAULT_LABELS
    raw_truth = frame_labels(args.image, args.image_root, label_file)
    # CNR-EXT stores free=0 and busy=1. Translate those into the model's indices.
    label_map = {0: mapping["free"], 1: mapping["busy"]}
    truth = {slot: label_map[label] for slot, label in raw_truth.items()}
    class_names = {mapping["free"]: "Vacant", mapping["busy"]: "Occupied"}
    disagreements = [first != second for first, second in zip(baseline, gnn)]
    left = draw_panel(image, "MobileNetV2 baseline", slots, baseline, disagreements, truth, class_names)
    right = draw_panel(image, "MobileNetV2 + GNN", slots, gnn, disagreements, truth, class_names)
    comparison = cv2.hconcat([left, right])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), comparison):
        raise OSError(f"Could not save comparison: {args.output}")

    print(f"Changed predictions: {sum(disagreements)} of {len(slots)} slots")
    if not any(int(space["id"]) in truth for space, *_ in slots):
        print("No matching ground-truth labels for this frame. Outlines show disagreements only.")
    for (space, *_), first, second in zip(slots, baseline, gnn):
        if first != second:
            actual = truth.get(int(space["id"]))
            print(f'Slot {space["id"]}: {class_names[first]} -> {class_names[second]}'
                  f' | truth: {class_names.get(actual, "unavailable")}')
    print(f"Saved comparison: {args.output}")

    if not args.no_display:
        cv2.namedWindow("Parking comparison", cv2.WINDOW_NORMAL)
        height, width = comparison.shape[:2]
        display_width = min(width, 1400)
        cv2.resizeWindow("Parking comparison", display_width, round(height * display_width / width))
        cv2.imshow("Parking comparison", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Full camera frame")
    parser.add_argument("--camera-csv", type=Path, required=True, help="CSV for this camera")
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--image-root", type=Path, default=detector.DATA_DIR,
                        help="Root with the weather folders for matching ground truth")
    parser.add_argument("--labels", type=Path, help="Label list for ground truth (defaults to test.txt)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    for path in (args.image, args.camera_csv, args.context, args.labels, detector.MODEL_PATH):
        if path is not None and not path.is_file():
            parser.error(f"File not found: {path}")
    try:
        run(args)
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"Parking comparison: {error}\n")
