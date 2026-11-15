import csv
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
"""
Dataset: CNRPark+EXT
Classes: free (0), busy (1)

Bounding box colors (OpenCV BGR format):
    Green  - Vacant
    Red    - Occupied
    Yellow - Uncertain

Note:
    This program uses predefined parking-space annotations.
    It does not automatically discover parking-space locations.
"""
# Resolve paths relative to this script instead of the terminal directory.
PROJECT_DIR = Path(__file__).resolve().parent
# Full parking lot images and camera-specific CSV annotations.
DATA_DIR = PROJECT_DIR / "dataset" / "CNR-EXT_FULL_IMAGE_1000x750" / "FULL_IMAGE_1000x750"
MODEL_PATH = PROJECT_DIR / "models" / "parking_classifier.pth"

# Location of the trained MobileNetV2 checkpoint.
OUTPUT_DIR = PROJECT_DIR / "output"

# Original image dimensions used by the CSV annotations.
# The downloaded full images are resized, so coordinates must be scaled.
ORIGINAL_WIDTH = 2592
ORIGINAL_HEIGHT = 1944

# Shrink each annotated bounding box around its center.
# This helps reduce interference from neighboring vehicles or objects.
CROP_SCALE = 0.85

# Use an NVIDIA GPU when CUDA is available; otherwise use the CPU.
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Apply the same preprocessing used during model training.
# MobileNetV2 expects RGB images normalized with ImageNet statistics.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model():
    model = models.mobilenet_v2(weights=None)

    # Replace the original classifier with a binary classification layer.
    model.classifier[1] = nn.Linear(
        model.last_channel, 2
    )

    # Load the trained weights onto the selected device.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
     # Move the model to GPU/CPU and disable training-specific behavior.
    model = model.to(device)
    model.eval()

    return model, checkpoint["class_to_idx"]


def load_parking_spaces(csv_path, image_width, image_height):
    spaces = []

     # Determine how much the annotation coordinates must be scaled.
    scale_x = image_width / ORIGINAL_WIDTH
    scale_y = image_height / ORIGINAL_HEIGHT

    with open(csv_path, newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
             # Convert the original coordinates to the current image size.
            x = int(float(row["X"]) * scale_x)
            y = int(float(row["Y"]) * scale_y)
            w = int(float(row["W"]) * scale_x)
            h = int(float(row["H"]) * scale_y)

            spaces.append({
                "id": row["SlotId"],
                "x": x,
                "y": y,
                "w": w,
                "h": h
            })

    return spaces


def detect_parking(image_path, csv_path):
    # Load the full parking lot image in OpenCV's BGR format.
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

     # Load the trained classifier and its class-label mapping.
    model, class_to_idx = load_model()

    height, width = image.shape[:2]

     # Retrieve parking-space coordinates for the selected camera.
    spaces = load_parking_spaces(
        csv_path, width, height
    )

    crops = []
    valid_spaces = []

    for space in spaces:
        x = space["x"]
        y = space["y"]
        w = space["w"]
        h = space["h"]

        # Find the center of the annotated bounding box.
        # Keeping the center fixed allows us to shrink the crop
        # without shifting it toward another parking space.
        center_x = x + w / 2
        center_y = y + h / 2

        # Reduce the crop dimensions
        new_w = w * CROP_SCALE
        new_h = h * CROP_SCALE

        # Calculate the new crop boundaries
        x1 = max(0, int(center_x - new_w / 2))
        y1 = max(0, int(center_y - new_h / 2))
        x2 = min(width, int(center_x + new_w / 2))
        y2 = min(height, int(center_y + new_h / 2))

        # Skip invalid or empty bounding boxes.
        if x2 <= x1 or y2 <= y1:
            continue

         # Extract the parking-space region from the full image.
        crop = image[y1:y2, x1:x2]

        rgb_crop = cv2.cvtColor(
            crop, cv2.COLOR_BGR2RGB
        )
        
        tensor = transform(
            Image.fromarray(rgb_crop)
        )

        # Store the tensor and its corresponding parking-space data.
        # The order must remain consistent for matching predictions.
        crops.append(tensor)
        valid_spaces.append((space, x1, y1, x2, y2))

    if not crops:
        raise ValueError("No valid parking-space crops found.")

    # Classify all parking spaces in one batch
    batch = torch.stack(crops).to(device)

    # Disable gradient tracking because we are only making predictions.
    with torch.inference_mode():
        outputs = model(batch)
        # Convert raw classification scores into probabilities.
        probabilities = torch.softmax(outputs, dim=1)

        # Select the highest-scoring class for each parking space.
        predictions = probabilities.argmax(dim=1).cpu().tolist()
        # Record the model's confidence in its selected class.
        confidences = probabilities.max(dim=1).values.cpu().tolist()

     # Use the checkpoint mapping instead of assuming class indices.
    free_label = class_to_idx["free"]

    # Predictions below this threshold are marked as uncertain.
    CONFIDENCE_THRESHOLD = 0.65

    free_count = 0
    occupied_count = 0
    uncertain_count = 0

    for (space, x1, y1, x2, y2), prediction, confidence in zip(
    valid_spaces, predictions, confidences
    ):
        if confidence < CONFIDENCE_THRESHOLD:
            color = (0, 255, 255)  # Yellow: uncertain
            uncertain_count += 1

        elif prediction == free_label:
            color = (0, 255, 0)    # Green: vacant
            free_count += 1

        else:
            color = (0, 0, 255)    # Red: occupied
            occupied_count += 1

         # Draw the classified parking-space bounding box.
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # Display the parking-space ID and prediction confidence.
        cv2.putText(
            image,
            f"{space['id']}: {confidence:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1
        )   
    # Display occupancy statistics
    cv2.rectangle(
        image, (10, 10), (320, 115),
        (0, 0, 0), -1
    )

    # Display confidently predicted available spaces.
    cv2.putText(
        image,
        f"Available: {free_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Display confidently predicted occupied spaces.
    cv2.putText(
        image,
        f"Occupied: {occupied_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    # Display parking spaces requiring further inspection.
    cv2.putText(
        image,
        f"Uncertain: {uncertain_count}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Create the output directory if it does not already exist.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "parking_detection.jpg"

    # Save the image with its color-coded predictions.
    cv2.imwrite(str(output_path), image)

    print(f"Available spaces: {free_count}")
    print(f"Occupied spaces: {occupied_count}")
    print(f"Uncertain spaces: {uncertain_count}")
    print(f"Total classified: {free_count + occupied_count + uncertain_count}")
    print(f"Saved result: {output_path}")

    cv2.imshow("Parking Occupancy Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

     # Select the weather condition, capture date, and camera.
    # Change these values to test different parking lot images.
    camera_dir = (
        DATA_DIR
        / "OVERCAST"
        / "2015-11-16"
        / "camera1"
    )

    # Retrieve JPG images and select the first one alphabetically.
    images = sorted(camera_dir.glob("*.jpg"))

    if not images:
        raise FileNotFoundError(
            f"No images found in: {camera_dir}"
        )

    image_path = images[0]

    # Camera annotations are stored one level above DATA_DIR.
    # The CSV must correspond to the camera used by the image.
    csv_path = DATA_DIR.parent / "camera1.csv"

     # Print path information to help diagnose missing files.
    print("Selected image:", image_path.name)
    print("Image exists:", image_path.exists())
    print("CSV exists:", csv_path.exists())

    detect_parking(image_path, csv_path)