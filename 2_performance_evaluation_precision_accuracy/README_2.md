# Airplane Detector Evaluation Challenge

## Background
Evaluating the performance of object detection models is a key skill in computer vision. This exercise focuses on assessing an airplane detector applied to satellite images. Using a dataset of images, I compared model predictions against human-annotated ground truth to compute standard metrics like precision and recall. The dataset includes images in the `planes/` directory, ground truth annotations in `groundtruth.json`, and model predictions with confidence scores in `predictions.json`.

## Objective
The goal is to evaluate the airplane detector’s performance by:
1. Computing true positives (TP), false positives (FP), and false negatives (FN) per image using Intersection over Union (IoU).
2. Calculating precision and recall across the entire dataset for multiple confidence thresholds.
3. Optimizing the evaluation process to reduce computation time while maintaining accuracy.

### Example
An example of airplane detection in a satellite image:
![Airplane Example](https://storage.googleapis.com/dp-missions/hiring-sa/plane.png)




## Solution: Airplane Detector Performance Evaluation

### Description
This project evaluates the precision and recall of an airplane detector on satellite images. The script `evaluate_perf.py` compares model predictions against ground truth annotations, computes per-image metrics using IoU, and calculates dataset-wide precision and recall for various thresholds, with an optimized implementation to enhance efficiency.

### Implemented Tasks
1. **Per-Image Evaluation**
   - Compares predicted boxes with ground truth using IoU ≥ 0.5.
   - Sorts predictions by confidence score (highest to lowest) to prioritize high-scoring matches.
   - Tracks matched ground truth boxes to avoid double counting.

2. **Precision and Recall Calculation**
   - Aggregates TP, FP, and FN across all images for thresholds from 0.0 to 1.0 (step size 0.1).
   - Computes precision and recall for the entire dataset at each threshold.

3. **Optimized Implementation**
   - Reduces unnecessary IoU calculations by pre-sorting predictions once and processing thresholds in descending order (1.0 to 0.0).

   - Implements early stopping when predictions fall below the threshold, leveraging a precomputed sorted list and efficient dictionary lookups.


### Requirements
- **Python:** 3.x
- **Libraries:** `shapely` (install via `pip install shapely`), `json` (built-in), `time` (built-in)

### Files Structure

├── README_2
├── evaluate_perf.py
├── groundtruth.json
├── planes
│   ├── pic1.png
│   ├── pic10.png
│   ├── pic11.png
│   ├── pic12.JPG
│   ├── pic13.png
│   ├── pic2.png
│   ├── pic3.png
│   ├── pic4.png
│   ├── pic5.JPG
│   ├── pic6.JPG
│   ├── pic7.JPG
│   ├── pic8.png
│   └── pic9.JPG
├── predictions.json
└── task3.ipynb


### Key Functions
- **`evaluate_image()`**
  - Computes TP, FP, FN for a single image, prioritizing high-confidence predictions.
- **`evaluate_pr_naive()`**
  - Baseline implementation for precision and recall across thresholds.
- **`evaluate_pr()`**
  - Optimized version reducing computation overhead.

### Usage
Run the evaluation:
```bash
python evaluate_perf.py
```

### Output
Per-Image Metrics: TP, FP, FN for each image at threshold 0.5.
Precision/Recall Curves: Metrics for thresholds [0.0, 1.0].
Performance Comparison: Naive vs. optimized runtime.



### Results
Naive version computed in 0.12s
Optimized version computed in 0.1s
Threshold 0.0 → Naive: P=0.93, R=0.89 | Optimized: P=0.93, R=0.89
Threshold 0.1 → Naive: P=0.94, R=0.89 | Optimized: P=0.94, R=0.89
Threshold 0.2 → Naive: P=0.96, R=0.89 | Optimized: P=0.96, R=0.89
Threshold 0.3 → Naive: P=1.00, R=0.89 | Optimized: P=1.00, R=0.89
Threshold 0.4 → Naive: P=1.00, R=0.86 | Optimized: P=1.00, R=0.86
Threshold 0.5 → Naive: P=1.00, R=0.77 | Optimized: P=1.00, R=0.77
Threshold 0.6 → Naive: P=1.00, R=0.68 | Optimized: P=1.00, R=0.68
Threshold 0.7 → Naive: P=1.00, R=0.56 | Optimized: P=1.00, R=0.56
Threshold 0.8 → Naive: P=1.00, R=0.37 | Optimized: P=1.00, R=0.37
Threshold 0.9 → Naive: P=1.00, R=0.21 | Optimized: P=1.00, R=0.21
Threshold 1.0 → Naive: P=1.00, R=0.16 | Optimized: P=1.00, R=0.16


### Performance Improvement:
- Naive version: 0.12s
- Optimized version: 0.10s

### Results Consistency:
Both implementations maintain identical precision and recall values across all thresholds
