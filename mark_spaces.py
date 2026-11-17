"""Mark parking spaces on a reference frame and export a detector-ready CSV."""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


# Match the coordinate system used by main.py's parking-space loader.
CSV_WIDTH = 2592
CSV_HEIGHT = 1944
WINDOW = "Mark parking spaces"


def bounds(corners, width, height):
    left = min(x for x, _ in corners)
    top = min(y for _, y in corners)
    right = max(x for x, _ in corners)
    bottom = max(y for _, y in corners)
    return (
        left * CSV_WIDTH / width,
        top * CSV_HEIGHT / height,
        (right - left) * CSV_WIDTH / width,
        (bottom - top) * CSV_HEIGHT / height,
    )


def save_csv(path, spaces, width, height):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects any layout already at this path.
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        corner_fields = [f"{axis}{i}" for i in range(1, 5) for axis in ("X", "Y")]
        writer.writerow(("SlotId", "X", "Y", "W", "H", *corner_fields))
        for slot_id, corners in enumerate(spaces, start=1):
            points = [
                value
                for x, y in corners
                for value in (x * CSV_WIDTH / width, y * CSV_HEIGHT / height)
            ]
            writer.writerow((slot_id, *bounds(corners, width, height), *points))


def mark_spaces(image, output):
    height, width = image.shape[:2]
    scale = min(1, 1200 / width, 700 / height)
    preview = cv2.resize(image, (max(1, round(width * scale)),
                                 max(1, round(height * scale))),
                         interpolation=cv2.INTER_AREA)
    preview_height, preview_width = preview.shape[:2]
    pending = []
    spaces = []
    message = "Click four corners in order around each space."
    controls = "U: undo    S: save and exit    Esc: cancel"

    def click(event, x, y, flags, data):
        nonlocal message
        if event != cv2.EVENT_LBUTTONDOWN or not (0 <= x < preview_width and 0 <= y < preview_height):
            return
        point = (x, y)
        if point in pending:
            message = "Choose a different corner."
            return
        pending.append(point)
        if len(pending) == 4:
            if not cv2.isContourConvex(np.array(pending, dtype=np.int32)):
                pending.pop()
                message = "Click around the perimeter in order. U: undo."
                return
            # Reject a zero-width or zero-height crop.
            _, _, box_width, box_height = bounds(pending, preview_width, preview_height)
            if box_width * width / CSV_WIDTH < 2 or box_height * height / CSV_HEIGHT < 2:
                pending.pop()
                message = "Box is too small. Pick a different fourth corner or undo."
                return
            spaces.append(pending.copy())
            pending.clear()
        message = f"{len(spaces)} space(s) marked. Current space: {len(pending)}/4 corners."

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.setMouseCallback(WINDOW, click)
        while True:
            display = cv2.copyMakeBorder(preview, 0, 55, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
            for slot_id, corners in enumerate(spaces, start=1):
                left = min(x for x, _ in corners)
                top = min(y for _, y in corners)
                polygon = np.array(corners, dtype=np.int32)
                cv2.polylines(display, [polygon], True, (0, 220, 0), 2)
            for point in pending:
                cv2.circle(display, point, 4, (0, 220, 255), -1)
            for start, end in zip(pending, pending[1:]):
                cv2.line(display, start, end, (0, 220, 255), 2)
            for offset, line in ((20, message), (43, controls)):
                cv2.putText(display, line, (8, preview_height + offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(30) & 0xFF

            if key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return False
            if key in (ord("u"), ord("U")):
                if not pending and spaces:
                    pending.extend(spaces.pop())
                if pending:
                    pending.pop()
                message = f"{len(spaces)} space(s) marked. Current space: {len(pending)}/4 corners."
            if key in (ord("s"), ord("S")):
                if pending or not spaces:
                    message = "Finish the current space and mark at least one before saving."
                    continue
                save_csv(output, spaces, preview_width, preview_height)
                print(f"Saved {len(spaces)} spaces to {output}")
                return True
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Reference frame from the video")
    parser.add_argument("--output", type=Path, required=True, help="New CSV to save")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Reference image not found: {args.image}")
    if args.output.exists():
        parser.error(f"Output already exists. Pick a new filename: {args.output}")
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"Could not read reference image: {args.image}")
    try:
        mark_spaces(image, args.output)
    except (OSError, cv2.error) as error:
        parser.exit(1, f"Could not mark parking spaces: {error}\n")
