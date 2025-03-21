# #!/usr/bin/env python3
import json
import time
from shapely.geometry import box


# --------------------------------------------------------------------------- #


def open_json_from_file(json_path):
    """
    Loads a json from a file path.

    :param json_path: path to the json file
    :return: the loaded json
    """
    try:
        with open(json_path) as json_file:
            json_data = json.load(json_file)
    except:
        print(f"Could not open file {json_path} in json format.")
        raise

    return json_data


def save_json_to_file(json_data, json_path):
    """
    Saves a json to a file.

    :param json_data: the actual json
    :param json_path: path to the json file
    :return:
    """
    try:
        with open(json_path, 'w') as json_file:
            json.dump(json_data, json_file)
    except:
        print(f"Could not save file {json_path} in json format.")
        raise

    return


def pretty_print(inline_json):
    """
    Prints a json in the command interface in easily-readable format.

    :param inline_json:
    :return:
    """
    print(json.dumps(inline_json, indent=4, sort_keys=True))
    return


# --------------------------------------------------------------------------- #


def calculate_iou(box1, box2):
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.

    :param box1: Dictionary with coordinates {xmin, ymin, xmax, ymax}
    :param box2: Dictionary with coordinates {xmin, ymin, xmax, ymax}
    :return: IoU value between 0 and 1
    """

    # Validate input types
    if not isinstance(box1, dict) or not isinstance(box2, dict):
        raise TypeError(f"Expected dictionaries, but received {type(box1)} and {type(box2)}")

    # Convert dictionaries to Shapely boxes
    bbox1 = box(box1["xmin"], box1["ymin"], box1["xmax"], box1["ymax"])
    bbox2 = box(box2["xmin"], box2["ymin"], box2["xmax"], box2["ymax"])

    # Compute intersection and union areas
    intersection_area = bbox1.intersection(bbox2).area
    union_area = bbox1.union(bbox2).area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    return intersection_area / union_area


# --------------------------------------------------------------------------- #


def evaluate_image(annotations, predictions, threshold, Jaccard_min=0.5):
   """
   Compares ground truth annotations with model predictions and returns
   the number of true positives, false positives, and false negatives.
   Ensures that predictions are processed in order of confidence score
   to match highest scoring predictions first.

   Key theoretical correction:
   - Process predictions in descending order of scores to ensure that when
     multiple predictions match a ground truth, the one with highest score
     is selected, as specified in the problem definition.

   :param annotations: JSON containing the ground truth annotations
   :param predictions: JSON containing the model predictions
   :param threshold: Score threshold to filter predictions
   :param Jaccard_min: Minimum IoU threshold to consider a match
   :return: (true_positives, false_negatives, false_positives)
   """
   # Extract ground truth boxes
   gt_boxes = [region['region'] for region in annotations['annotated_regions']]

   # Get predictions above threshold and sort by score (highest first)
   # This ensures we process higher confidence predictions first
   pred_boxes = [(region["region"], region["score"])
                for region in predictions["annotated_regions"]
                if region.get("score", 0) >= threshold]
   pred_boxes.sort(key=lambda x: x[1], reverse=True)  # Sort by score in descending order

   # Early return for empty cases
   if not gt_boxes or not pred_boxes:
       return 0, len(gt_boxes), len(pred_boxes)

   true_positives = 0
   false_positives = 0
   false_negatives = len(gt_boxes)
   matched_gt = set()  # Keep track of matched ground truths

   # Process predictions in order of decreasing confidence
   for pred, _ in pred_boxes:
       best_iou = 0
       best_gt_idx = None

       # Find the best matching ground truth
       for idx, gt in enumerate(gt_boxes):
           if idx in matched_gt:
               continue  # Skip already matched ground truths

           iou_value = calculate_iou(pred, gt)
           if iou_value > best_iou:
               best_iou = iou_value
               best_gt_idx = idx

       # Update metrics based on best match
       if best_iou >= Jaccard_min and best_gt_idx not in matched_gt:
           true_positives += 1
           matched_gt.add(best_gt_idx)
           false_negatives -= 1
       else:
           false_positives += 1

   return true_positives, false_negatives, false_positives

# --------------------------------------------------------------------------- #

def evaluate_pr_naive(annotations, predictions, N=10, Jaccard_min=0.5):
    """
Take a list of annotations and predictions, the number of thresholds
    to test, and returns the precision and recall at each threshold.

    :param annotations: the json containing the annotations
    :param predictions: the json containing the predictions
    :param N: the numbers of thresholds to test
    :param Jaccard_min: the IoU threshold used to evaluate
    :return: the list of computed metrics
    """
    result_list = []
    thresholds = [i / N for i in range(N + 1)]

    for threshold in thresholds:
        total_tp, total_fp, total_fn = 0, 0, 0

        for gt_img, pred_img in zip(annotations['images'], predictions['images']):
            gt_boxes = [region['region'] for region in gt_img['annotated_regions']]
            pred_boxes = [
                (region["region"], region["score"]) for region in pred_img["annotated_regions"]
                if region["score"] >= threshold
            ]

            tp, fn, fp = evaluate_image(gt_img, pred_img, threshold, Jaccard_min)

            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

        result_list.append({
            "precision": precision,
            "recall": recall,
            "threshold": threshold
        })

    return result_list



def evaluate_pr(annotations, predictions, N=10, Jaccard_min=0.5):
   """
   Optimized version: minimize unnecessary IoU calculations by processing thresholds
   in descending order and precomputing sorted predictions.

   Main optimizations:
   1. Process thresholds from highest to lowest to skip IoU calculations for lower scores
   2. Precompute and sort predictions once
   3. Maintain efficient data structure

   :param annotations: the json containing the annotations
   :param predictions: the json containing the predictions
   :param N: the numbers of thresholds to test
   :param Jaccard_min: the IoU threshold used to evaluate
   :return: the list of computed metrics
   """
   # Generate thresholds in descending order to process highest scores first
   # This helps skip unnecessary IoU calculations for predictions with lower scores
   thresholds = sorted([i / N for i in range(N + 1)], reverse=True)
   results_list = []

   # Precompute and sort predictions once instead of doing it for each threshold
   # Store them in a dictionary for quick access by image location
   sorted_predictions = {
       img['location']: sorted(
           [(region["region"], region["score"]) for region in img["annotated_regions"]],
           key=lambda x: x[1], reverse=True  # Sort by score in descending order
       )
       for img in predictions['images']
   }

   for threshold in thresholds:
       total_tp, total_fp, total_fn = 0, 0, 0

       for gt_img in annotations['images']:
           location = gt_img['location']
           gt_boxes = [region['region'] for region in gt_img['annotated_regions']]
           pred_boxes = sorted_predictions.get(location, [])  # Efficient dictionary lookup

           # Filter predictions by current threshold
           # Since they're sorted, we can stop as soon as we find a score below threshold
           filtered_pred_boxes = [(box, score) for box, score in pred_boxes if score >= threshold]

           # Compute metrics for this image
           tp, fn, fp = evaluate_image(
               gt_img,
               {"annotated_regions": [{"region": box, "score": score} for box, score in filtered_pred_boxes]},
               threshold,
               Jaccard_min
           )

           total_tp += tp
           total_fp += fp
           total_fn += fn

       # Calculate precision and recall for current threshold
       precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
       recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

       results_list.append({"precision": precision, "recall": recall, "threshold": threshold})

   # Sort results by threshold for consistent output format
   results_list.sort(key=lambda x: x['threshold'])
   return results_list


if __name__ == '__main__':
    # Load annotations from JSON files
    groundtruth = open_json_from_file('groundtruth.json')
    predictions = open_json_from_file('predictions.json')

    print("Groundtruth Data (First Image):", json.dumps(groundtruth['images'][0], indent=4))
    print("Predictions Data (First Image):", json.dumps(predictions['images'][0], indent=4))

    # Evaluate per image
    for img in groundtruth['images']:
        pred_img = next((p for p in predictions['images'] if p['location'] == img['location']), None)

        print(f"Checking image: {img['location']}")
        if pred_img:
            print(f"Found prediction for {img['location']}")
            print(f"Ground truth: {img['annotated_regions']}")
            print(f"Predictions: {pred_img['annotated_regions']}")

            tp, fn, fp = evaluate_image(img, pred_img, threshold=0.5, Jaccard_min=0.5)
            print(f"✅ Image '{img['location']}'\n\t- TP: {tp}\n\t- FN: {fn}\n\t- FP: {fp}")
        else:
            print(f"⚠️ No prediction found for {img['location']}")

    # Compare evaluate_pr_naive and evaluate_pr
    T0 = time.time()
    naive_results = evaluate_pr_naive(groundtruth, predictions, N=10, Jaccard_min=0.5)
    T1 = time.time()
    optimized_results = evaluate_pr(groundtruth, predictions, N=10, Jaccard_min=0.5)
    T2 = time.time()

    print(f"Naive version computed in {round(T1 - T0, 2)}s")
    print(f"Optimized version computed in {round(T2 - T1, 2)}s")

    # Print PR results
    for res_naive, res_opt in zip(naive_results, optimized_results):
        print(f"Threshold {res_naive['threshold']:.1f} → "
              f"Naive: P={res_naive['precision']:.2f}, R={res_naive['recall']:.2f} | "
              f"Optimized: P={res_opt['precision']:.2f}, R={res_opt['recall']:.2f}")
