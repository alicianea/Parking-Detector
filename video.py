"""Show GNN parking occupancy on a video or live camera."""
import argparse
import math
from pathlib import Path
from time import perf_counter

import cv2
import torch

import main as detector
from slot_context import load_context, slot_geometry


@torch.inference_mode()
def classify_frame(frame, camera_csv, baseline, context, free_label):
    # Use the same crops and slot geometry as the still-image detector.
    crops, spaces = detector.extract_frame_slots(frame, camera_csv)
    geometry = slot_geometry(spaces, frame.shape).to(detector.device)
    features, original_scores = context.encode(baseline, crops.to(detector.device))
    scores = context(features, original_scores, geometry)
    confidence, labels = scores.softmax(dim=1).max(dim=1)

    results = []
    for (space, x1, y1, x2, y2), label, score in zip(
        spaces, labels.tolist(), confidence.tolist()
    ):
        if score < 0.65:
            status = "Uncertain"
        elif label == free_label:
            status = "Vacant"
        else:
            status = "Occupied"
        results.append((space["id"], (x1, y1, x2, y2), space.get("polygon"), status, score))
    return results


def draw_results(frame, results):
    colors = {"Vacant": (0, 255, 0), "Occupied": (0, 0, 255), "Uncertain": (0, 255, 255)}
    counts = dict.fromkeys(colors, 0)
    for slot_id, box, polygon, status, confidence in results:
        x1, y1, _, _ = box
        counts[status] += 1
        color = colors[status]
        detector.draw_slot_outline(frame, box, polygon, color)
        cv2.putText(frame, f"{slot_id}: {confidence:.2f}", (x1, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    summary = "  |  ".join(f"{status}: {count}" for status, count in counts.items())
    (width, height), _ = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (0, 0), (width + 16, height + 16), (0, 0, 0), -1)
    cv2.putText(frame, summary, (8, height + 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)


def run_video(source, camera_csv, checkpoint, every):
    baseline, mapping = detector.load_model()
    context = load_context(checkpoint, detector.MODEL_PATH, detector.device)
    capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        # Recordings target their original frame rate. Live cameras pace reads themselves.
        frame_period = 0
        if isinstance(source, str):
            frame_period = 1 / fps if math.isfinite(fps) and fps > 0 else 1 / 30
        frame_number = 0
        previous_size = None
        print(f"Running on {detector.device}. Press Q or Esc to stop.")
        while True:
            started = perf_counter()
            ok, frame = capture.read()
            if not ok:
                if frame_number == 0:
                    raise ValueError(f"Could not read the first video frame: {source}")
                break

            size = frame.shape[:2]
            # Keep the latest predictions visible between inference frames.
            if size != previous_size or frame_number % every == 0:
                results = classify_frame(frame, camera_csv, baseline, context, mapping["free"])
                previous_size = size
            draw_results(frame, results)
            cv2.imshow("Parking Detector", frame)
            frame_number += 1

            remaining = frame_period - (perf_counter() - started)
            key = cv2.waitKey(max(1, round(1000 * remaining))) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty("Parking Detector", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Path to a recorded video")
    source.add_argument("--camera", type=int, help="Live camera index, usually 0")
    parser.add_argument("--camera-csv", type=Path, required=True,
                        help="Slot annotations for this exact camera view")
    parser.add_argument("--context", type=Path,
                        default=detector.PROJECT_DIR / "models" / "slot_context.pth")
    parser.add_argument("--every", type=int, default=5, metavar="N",
                        help="Update predictions every N frames (default: 5)")
    args = parser.parse_args()
    if args.every < 1:
        parser.error("--every must be at least 1")
    if args.camera is not None and args.camera < 0:
        parser.error("--camera must be nonnegative")
    for path in (args.video, args.camera_csv, detector.MODEL_PATH, args.context):
        if path is not None and not path.is_file():
            parser.error(f"File not found: {path}")
    try:
        run_video(str(args.video) if args.video is not None else args.camera,
                  args.camera_csv, args.context, args.every)
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"Parking Detector: {error}\n")