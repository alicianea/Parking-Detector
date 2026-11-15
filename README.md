# Parking Occupancy Detector

A computer vision project that uses deep learning to identify vacant and occupied parking spaces in parking lot images.

The application uses a MobileNetV2 convolutional neural network (CNN) trained through transfer learning to classify individual parking spaces. OpenCV is used to visualize the predictions on full parking lot images.

## Features

- Classifies parking spaces as vacant or occupied.
- Processes full parking lot images using camera-specific annotations.
- Draws color-coded bounding boxes:
  - **Green:** Vacant
  - **Red:** Occupied
  - **Yellow:** Uncertain
- Displays the number of available, occupied, and uncertain spaces.
- Saves annotated images for review.
- Supports GPU acceleration through CUDA when available.

## Technologies

- Python
- PyTorch and Torchvision
- MobileNetV2
- OpenCV
- Pillow

## Dataset

This project uses the **CNRPark+EXT** dataset, which contains parking lot images captured under different weather and lighting conditions.

Two dataset components are used:

**CNR-EXT-Patches-150x150**

Contains individual parking-space images used for training, validation, and testing.

**CNR-EXT_FULL_IMAGE_1000x750**

Contains full parking lot images and camera-specific CSV annotations used for occupancy detection.

The dataset is not included in this repository and must be downloaded separately.

## Model Architecture

The classifier uses MobileNetV2 with pretrained ImageNet weights.

The original classification layer is replaced with a two-class output layer:

- `0` — Free
- `1` — Occupied

During initial training, the feature extraction layers are frozen while the classification layer is trained on parking-space images.

Input images are resized to 224 × 224 pixels and normalized using ImageNet statistics.

## Model Evaluation

The trained classifier achieved the following results on the patch dataset's test split:

| Metric | Result |
|---|---|
| Accuracy | 98.14% |
| Precision | 98.77% |
| Recall | 97.98% |
| F1 Score | 98.37% |

These results measure performance on individual parking-space patches, not the complete full-image detection pipeline.

## Installation

Clone the repository:

```bash
git clone https://github.com/alicianea/Parking-Detector.git
cd Parking-Detector
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Download the CNRPark+EXT dataset and place the extracted files in the appropriate directories under `dataset/`.

## Usage

### Train the classifier

```bash
python train.py
```

This trains the MobileNetV2 classifier and saves its checkpoint to:

`models/parking_classifier.pth`

### Evaluate the classifier

```bash
python evaluate.py
```

This evaluates the trained classifier using the test dataset.

### Run full-image parking detection

```bash
python main.py
```

The detector loads a parking lot image, extracts parking-space regions using camera annotations, and classifies each space.

The annotated image is saved to:

`output/parking_detection.jpg`

The weather condition, date, and camera can currently be changed in the `main.py` entry point.

**Note:** Full-image detection requires a trained model checkpoint. The checkpoint is not included in the repository.

## Detection Configuration

Two parameters can be adjusted in `main.py`:

- `CROP_SCALE = 0.85`: Reduces the size of parking-space crops to limit interference from neighboring objects.
- `CONFIDENCE_THRESHOLD = 0.65`: Marks predictions below the threshold as uncertain.

These values were selected through preliminary visual testing and may require adjustment for different cameras or environmental conditions.

## Current Limitations

- Parking-space locations are provided through camera-specific CSV annotations rather than automatically detected.
- Trees, parked vehicles, and other obstructions can interfere with classification.
- Distant parking spaces may be difficult to classify.
- Model performance may vary across different weather conditions and camera viewpoints.
- Confidence scores are not formally calibrated.

## Future Improvements

- Evaluate full-image detection across multiple weather conditions and cameras.
- Improve classification of distant and partially obscured spaces.
- Experiment with fine-tuning additional MobileNetV2 layers.
- Extend detection to video frames.
- Investigate automatic parking-space localization.
