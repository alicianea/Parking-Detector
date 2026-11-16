# Parking Detector

Parking Detector aims to improve parking occupancy detection when an
individual space is hard to classify. For example, part of a vehicle in a
neighboring space can appear inside a vacant space’s image crop and make
that space look occupied.

The project builds on a MobileNetV2 baseline that classifies each marked
space independently in fixed-camera images from CNR-EXT. A graph neural
network (GNN) refines those predictions using visual clues from nearby
spaces in the same frame. We compare the baseline, a refinement model
that uses slot geometry, and the GNN on the same held-out spaces to
measure how much neighbor information helps.

## Project layout

Run the commands below from the repository root. Set up the
[CNR-EXT dataset](https://cnrpark.it/) and the provided checkpoints like this:

```text
parking-detector/
├── README.md
├── main.py                      # Run the detector on one image
├── parking_dataset.py           # Load labeled parking-space crops
├── train.py                     # Train the MobileNetV2 baseline
├── slot_context.py              # Define the GNN and load its checkpoint
├── train_context.py             # Train or evaluate the refinement models
├── evaluate.py                  # Evaluate MobileNetV2 on pre-cropped images
├── models/
│   ├── parking_classifier.pth   # Matching MobileNetV2 baseline
│   ├── slot_context.pth         # GNN refinement
│   └── slot_geometry.pth        # Refinement without neighbor messages
├── dataset/
│   ├── CNR-EXT-Patches-150x150/
│   │   ├── LABELS/
│   │   │   ├── train.txt
│   │   │   ├── val.txt
│   │   │   └── test.txt
│   │   └── PATCHES/             # Needed for train.py and evaluate.py
│   └── CNR-EXT_FULL_IMAGE_1000x750/
│       ├── camera1.csv … camera9.csv
│       └── FULL_IMAGE_1000x750/
│           ├── SUNNY/
│           ├── OVERCAST/
│           └── RAINY/
└── output/                      # Created by the scripts below
```

Keep the archive’s date and camera subfolders under each weather folder.
Install PyTorch, torchvision, Pillow, and the desktop OpenCV package in your
Python environment if they are missing:

```powershell
python -m pip install torch torchvision pillow opencv-python
```

**You can run the demo and weather evaluation without retraining.** Keep the
provided `parking_classifier.pth` paired with the two refinement checkpoints.
Each refinement checkpoint verifies the exact MobileNetV2 file it was trained
against. If you have `parking_classifier_fast.pth`, it is a separate baseline
experiment.

## Run the image demo

```powershell
python main.py --context models/slot_context.pth
```

The annotated image is saved to `output/parking_detection.jpg`. To select a
different frame, add `--image "path/to/frame.jpg" --camera-csv "path/to/camera1.csv"`
and choose the CSV for the frame’s camera. Add `--no-display` to save the result
without opening a window.

## Evaluate by weather

The weather evaluation needs the patch archive’s **label lists** and the full
images with their camera CSVs. It crops each parking space directly from the
full image, so you can skip extracting `PATCHES/` for this task.

Check the dataset paths and slot IDs before running inference:

```powershell
python train_context.py --check-data
```

### Create the test subsets

In PowerShell, create three subsets by copying the corresponding rows directly
from `test.txt`. The source test list stays intact.

```powershell
$test = "dataset\CNR-EXT-Patches-150x150\LABELS\test.txt"
$subsetDir = "output\weather"
$conditions = @("SUNNY", "OVERCAST", "RAINY")
New-Item -ItemType Directory -Force $subsetDir | Out-Null

foreach ($weather in $conditions) {
    $rows = @(Get-Content -LiteralPath $test | Where-Object { $_ -match "^$weather/" })
    if ($rows.Count -eq 0) {
        throw "No held-out labels found for $weather in $test"
    }
    $subset = Join-Path $subsetDir "$weather.txt"
    Set-Content -LiteralPath $subset -Value $rows -Encoding ascii
    Write-Host "$weather - $($rows.Count) labeled test slots"
}
```

### Run the models

Run the GNN and the geometry-only checkpoint on each subset in the same PowerShell session.

```powershell
foreach ($weather in $conditions) {
    $subset = Join-Path $subsetDir "$weather.txt"
    Write-Host "`n$weather - GNN"
    python train_context.py --evaluate --checkpoint models/slot_context.pth --subset $subset
    if ($LASTEXITCODE -ne 0) { throw "GNN evaluation failed for $weather" }

    Write-Host "`n$weather - geometry only"
    python train_context.py --evaluate --checkpoint models/slot_geometry.pth --subset $subset
    if ($LASTEXITCODE -ne 0) { throw "Geometry-only evaluation failed for $weather" }
}
```

Each command also prints MobileNetV2 results on the same slots and crops.
The MobileNetV2 result should agree between the two runs for each condition.
The geometry-only checkpoint uses the space’s own visual features and
coordinates. The GNN checkpoint also uses nearby spaces. Use the
`train_context.py` MobileNetV2 results for this comparison. The standalone
`evaluate.py` reads pre-cropped test images, which can produce different scores.

### Report the results

Fill in the number of matched frames, labeled slots, accuracy, and
occupied-class F1 for each condition. Save the four confusion-matrix counts
(TN, FP, FN, TP) for each model so you can see which errors changed.

| Weather | Frames | Labeled slots | MobileNetV2 accuracy / F1 | Geometry-only accuracy / F1 | GNN accuracy / F1 |
| --- | ---: | ---: | --- | --- | --- |
| Sunny | | | | | |
| Overcast | | | | | |
| Rainy | | | | | |

Compare the GNN with the geometry-only model within each weather condition to
see where neighbor information helped. Report the sample counts alongside the
scores so readers can see how much data each condition represents.

## Training and full-test evaluation

`train.py` trains MobileNetV2. `train_context.py` uses a frozen MobileNetV2
checkpoint and caches full-frame crop features before training the refinement
head. Keep a new baseline and the heads trained against it together when
running a separate experiment.

To evaluate the provided checkpoints on the entire held-out test split:

```powershell
python train_context.py --evaluate --checkpoint models/slot_context.pth
python train_context.py --evaluate --checkpoint models/slot_geometry.pth
```

To evaluate a manually tagged occlusion subset, pass
`--subset path/to/occluded_test.txt` with the relevant rows copied from
`test.txt`. The graph still uses all valid slots in each selected frame.
