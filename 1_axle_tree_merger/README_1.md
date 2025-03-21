# Axle Tree Merging Challenge

## Background
In computer vision, preprocessing data is a critical step before training object detection models. This exercise focuses on enhancing a vehicle detection dataset by merging closely spaced axle annotations. The dataset, derived from video frames, originally includes images annotated with three categories: `single_axle`, `car`, and `other`, stored in `annotations.json`.

## Objective
The goal is to transform `annotations.json` by merging nearby `single_axle` annotations into `grouped_axles` based on a proximity criterion, to represent sets of adjacent single axles —a common requirement in real world vehicle detection tasks-, while preserving the original structure and restricting the output to four tags: `car`, `other`, `single_axle`, and `grouped_axles`. This preprocessing step prepares the data for training a detection model capable of recognizing both individual and grouped axles.

### Example Transformation
- **Before:** Two separate `single_axle` annotations.
  ![Single Axles](https://storage.googleapis.com/dp-missions/hiring-sa/single.jpg)
- **After:** Merged into a single `grouped_axles` annotation (purple box).
  ![Grouped Axles](https://storage.googleapis.com/dp-missions/hiring-sa/grouped.jpg)

## Solution: Automatic Axle Tree Merger

### Description
This project implements an **automatic merging system** for single axle annotations in vehicle images. The script `merge_axle_tree.py` analyzes the proximity of `single_axle` bounding boxes along the X-axis and combines them into `grouped_axles` when they are sufficiently close, outputting the result to `new_annotations.json`. This preprocessing enhances the dataset for training a detection network capable of identifying both individual and grouped axles.

### Implemented Tasks
1. **Proximity Analysis**
   - Defined a proximity criterion: Two `single_axle` boxes are merged if the distance between one’s `xmin` and the other’s `xmax` is ≤ 0.01 (configurable threshold).
   - Boxes are sorted by `xmin` to ensure left-to-right processing.

2. **Annotation Transformation**
   - Extracts `single_axle` annotations from `annotations.json`.
   - Merges qualifying boxes into `grouped_axles` while preserving `car` and `other` annotations.
   - Generates a new JSON with updated regions.

3. **Format Validation**
   - Ensures the output JSON adheres to the required structure, containing only `car`, `other`, `single_axle`, and `grouped_axles`.

### Requirements
- **Python:** 3.x
- **Libraries:** `json` (built-in)

### Files Structure
├── README_1
├── annotations.json
├── axle_tree2.ipynb
├── images
│   ├── pic1.jpg
│   ├── pic2.jpg
│   ├── pic3.jpg
│   ├── pic4.jpg
│   └── pic5.jpg
├── merge_axle_tree.py
└── new_annotations.json

### Key Functions
- **`merge_boxes()`**
  - Sorts boxes by `xmin` and iteratively merges those within the threshold.
  - Returns a dictionary with `grouped_axles` and remaining `single_axle` boxes.
- **`update_annotations()`**
  - Updates the JSON structure, integrating merged and non-merged boxes.
- **`merge_axle_trees()`**
  - Orchestrates the process: extraction, merging, and updating.

### Usage
Run the transformation:
```bash
python merge_axle_tree.py
```

### Output
New_annotations.json with merged axle trees
Preservation of non-axle annotations
Validated tag structure (car, other, single_axle, grouped_axles)


### Results

Sample transformation results showing successful merging:

Image: images/pic1.jpg
Original single_axles: 6
Created grouped_axles: 2
Remaining single_axles: 2

Image: images/pic2.jpg
Original single_axles: 5
Created grouped_axles: 2
Remaining single_axles: 1
