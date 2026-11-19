# Parking occupancy evaluation

## Protocol

We used the official CNR-EXT train, validation, and test lists and checked that capture dates are disjoint across the combined train/validation set and test set. All nine test cameras also occur in training or validation. Each model scored the same labeled slots cropped from the original test frames. MobileNetV2 used the paired `parking_classifier.pth` checkpoint. Both refinement heads were evaluated as supplied. The geometry-only head uses the slot’s own visual features, baseline logits, and coordinates. The GNN also receives messages from four nearby slots when available. The evaluation used argmax for all labeled slots. Model weights were read without training.

The protocol uses the [official CNRPark+EXT dataset](http://cnrpark.it/).

## Results

| Split | Frames | Labeled slots | MobileNetV2 accuracy / F1 | Geometry-only accuracy / F1 | GNN accuracy / F1 |
| --- | ---: | ---: | --- | --- | --- |
| FULL | 896 | 31825 | 94.02% / 94.94% | 98.40% / 98.62% | 98.88% / 99.03% |
| SUNNY | 354 | 12600 | 92.06% / 90.40% | 97.99% / 97.45% | 99.19% / 98.96% |
| OVERCAST | 179 | 6365 | 96.12% / 97.37% | 99.45% / 99.62% | 99.48% / 99.65% |
| RAINY | 363 | 12860 | 94.90% / 96.31% | 98.29% / 98.74% | 98.28% / 98.74% |

### Confusion counts

| Split | Model | TN | FP | FN | TP |
| --- | --- | ---: | ---: | ---: | ---: |
| FULL | MobileNetV2 | 12064 | 1485 | 419 | 17857 |
| FULL | Geometry-only | 13186 | 363 | 145 | 18131 |
| FULL | GNN | 13331 | 218 | 138 | 18138 |
| SUNNY | MobileNetV2 | 6884 | 840 | 161 | 4715 |
| SUNNY | Geometry-only | 7504 | 220 | 33 | 4843 |
| SUNNY | GNN | 7666 | 58 | 44 | 4832 |
| OVERCAST | MobileNetV2 | 1544 | 164 | 83 | 4574 |
| OVERCAST | Geometry-only | 1688 | 20 | 15 | 4642 |
| OVERCAST | GNN | 1690 | 18 | 15 | 4642 |
| RAINY | MobileNetV2 | 3636 | 481 | 175 | 8568 |
| RAINY | Geometry-only | 3994 | 123 | 97 | 8646 |
| RAINY | GNN | 3975 | 142 | 79 | 8664 |

### Error changes

On the full test split, the GNN changed accuracy by +4.86 percentage points relative to MobileNetV2 and +0.48 points relative to geometry-only. Its occupied-class F1 changed by +4.09 and +0.41 points, respectively.

| Split | Baseline errors corrected | New errors vs baseline | Geometry errors corrected | New errors vs geometry |
| --- | ---: | ---: | ---: | ---: | ---: |
| FULL | 1681 | 133 | 315 | 163 |
| SUNNY | 938 | 39 | 192 | 41 |
| OVERCAST | 229 | 15 | 19 | 17 |
| RAINY | 514 | 79 | 104 | 105 |

Most of the improvement over MobileNetV2 was already present in the geometry-only checkpoint. The GNN reduced errors from 508 to 356 relative to geometry-only, with 315 corrections and 163 new errors. The 152 net corrections came almost entirely from SUNNY, with 151 fewer errors. OVERCAST had two fewer errors and RAINY had one more error. On SUNNY, the GNN reduced false positives from 220 to 58 while false negatives rose from 33 to 44. On RAINY it reduced false negatives from 97 to 79 while false positives rose from 123 to 142.

## Presentation examples

The cases are the first matches in frame-path and numeric slot order. A yellow outline marks the slot being scored.

| Category | Frame and slot | Ground truth | MobileNetV2 | Geometry-only | GNN |
| --- | --- | --- | --- | --- | --- |
| [Baseline error corrected](baseline_fixed.png) | `OVERCAST/2015-12-03/camera1/2015-12-03_0741.jpg`, slot 195 | Vacant | Occupied | Vacant | Vacant |
| [New error vs baseline](new_vs_baseline.png) | `OVERCAST/2015-12-03/camera1/2015-12-03_0911.jpg`, slot 193 | Vacant | Vacant | Occupied | Occupied |
| [Geometry-only error corrected](geometry_fixed.png) | `OVERCAST/2015-12-03/camera1/2015-12-03_0911.jpg`, slot 222 | Occupied | Occupied | Vacant | Occupied |
| [New error vs geometry](new_vs_geometry.png) | `OVERCAST/2015-12-03/camera1/2015-12-03_0911.jpg`, slot 224 | Occupied | Occupied | Occupied | Vacant |

[Incoming-neighbor diagram](gnn_neighbors.png)

## Interpretation limits

These measurements describe the supplied checkpoints on this dataset split. The weather groups differ in cameras, capture dates, and occupied-slot prevalence. The comparisons do not isolate a causal weather effect, establish statistical significance, measure occlusion-specific performance, or test transfer to unseen cameras or lots. The example pictures illustrate observed errors without identifying which neighbor caused a correction.

## Reproduction

```bash
python3 train_context.py --check-data
python3 evaluate_report.py --encoder-device mps --output output/held_out_repeat
```

The run used Apple MPS for the baseline and CPU for both heads. See [manifest.json](manifest.json) for exact inputs and versions and [predictions.csv](predictions.csv) for the slot-level audit trail.
