# Parking Occupancy Detector

A computer vision project that uses a convolutional neural network (CNN) to classify parking spaces as vacant or occupied.

## Project Overview

This project aims to develop a parking occupancy detection system capable of identifying available parking spaces from camera images. The long-term goal is to support full parking lot images and video, with green outlines indicating vacant spaces and red outlines indicating occupied spaces.

## Current Implementation

- MobileNetV2-based binary parking occupancy classifier
- Transfer learning using pretrained ImageNet weights
- Custom PyTorch dataset loader for CNR-EXT
- Training and validation using the dataset's provided label files
- Model checkpoint saving for future inference

## Dataset

The project uses the CNRPark+EXT dataset.

Download the [CNR-EXT parking patches](https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNR-EXT-Patches-150x150.zip).

Extract the dataset into:

`dataset/CNR-EXT-Patches-150x150/`

The dataset is excluded from GitHub because of its size.

## Installation

Install the required Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Training

Run:

```bash
python train.py
```

The trained model is saved to:

`models/parking_classifier.pth`

## Initial Results

| Metric | Result |
|---|---|
| Model | MobileNetV2 |
| Training images | 94,493 |
| Validation images | 18,647 |
| Epochs | 2 |
| Best validation accuracy | 97.62% |
| Final validation accuracy | 97.50% |

These are preliminary validation results. Independent test-set evaluation is still pending.

## Planned Features

- Evaluate classification performance on the test dataset
- Detect occupancy in full parking lot images using predefined parking-space coordinates
- Draw green outlines for vacant spaces and red outlines for occupied spaces
- Extend occupancy detection to video frames
- Evaluate performance under different weather and lighting conditions
