# #!/usr/bin/env python3
import json


# -------------------------------------------------------------------------------------------------------------------- #


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


# -------------------------------------------------------------------------------------------------------------------- #





def merge_boxes(single_axle_boxes_by_image, overlap_threshold=0.01):
    """
    Merges bounding boxes of single_axle that are overlapping or very close in the X-axis.
    Returns a dictionary with merged grouped_axles and remaining single_axle boxes.

    :param single_axle_boxes_by_image: Dictionary where keys are image IDs and values are lists of bounding boxes
                                       (each box is a dict with 'xmin', 'ymin', 'xmax', 'ymax').
    :param overlap_threshold: A float that defines how close two bounding boxes must be on the X-axis to be merged.
                              If the distance between a box's xmin and the previous xmax is within this threshold,
                              they are merged.
    :return: Dictionary {image_id: {'grouped_axles': [...], 'single_axle': [...]}} where:
             - 'grouped_axles' contains merged bounding boxes.
             - 'single_axle' contains remaining non-merged boxes.
    """
    merged_boxes_by_image = {}


    # single_axle_boxes_by_image = {} # Extract single_axle bounding boxes for each image
    # for image in annotation["images"]:
    #    image_id = image["location"]
    #    single_axle_boxes_by_image[image_id] = [
    #        region["region"] for region in image["annotated_regions"] if "single_axle" in region["tags"]

    # Almacena todas las cajas de single_axle, organizadas por cada imagen.
    # La clave será el nombre del archivo de la imagen.
    # El valor será una lista con todas las coordenadas de single_axle de esa imagen.

    for image_id, single_axle_boxes in single_axle_boxes_by_image.items():
        merged_boxes = []
        remaining_single_axle = set(range(len(single_axle_boxes)))  # Track indices of non-merged single_axles

        # Sort bounding boxes by 'xmin' to process them in left-to-right order
        single_axle_boxes = sorted(single_axle_boxes, key=lambda b: b["xmin"])

        for i, box1 in enumerate(single_axle_boxes):
            if i not in remaining_single_axle:
                continue  # Skip if already merged

            group = [box1]  # Start a new group with the current box
            current_xmax = box1["xmax"]  # Track the rightmost boundary of the merged group

            for j in range(i + 1, len(single_axle_boxes)):
                box2 = single_axle_boxes[j]
                if j not in remaining_single_axle:
                    continue  # Skip if already merged

                # If box2 is close enough to box1 (or overlaps), merge them
                if box2["xmin"] <= current_xmax + overlap_threshold:
                    group.append(box2)
                    remaining_single_axle.remove(j)  # Mark box2 as merged
                    current_xmax = max(current_xmax, box2["xmax"])  # Update rightmost boundary

            # If more than one box was grouped together, create a grouped_axles bounding box
            if len(group) > 1:
                remaining_single_axle.remove(i)  # Mark box1 as merged
                merged_boxes.append({
                    "xmin": min(b["xmin"] for b in group),
                    "ymin": min(b["ymin"] for b in group),
                    "xmax": max(b["xmax"] for b in group),
                    "ymax": max(b["ymax"] for b in group)
                })

        # Store results: merged grouped_axles and remaining single_axles
        merged_boxes_by_image[image_id] = {
            "grouped_axles": merged_boxes,
            "single_axle": [single_axle_boxes[i] for i in remaining_single_axle]
        }

    return merged_boxes_by_image






def update_annotations(annotation, merged_boxes_by_image):
    """
    Updates the original JSON annotations by adding grouped_axles while keeping
    single_axle boxes that were not merged.

    :param annotation: Dictionary containing the original annotations from the JSON file.
                       It follows the format {'images': [{...}]}, where each image contains
                       'annotated_regions' with labeled bounding boxes.
    :param merged_boxes_by_image: Dictionary {image_id: {'grouped_axles': [...], 'single_axle': [...]}}.
                                  - 'grouped_axles' contains newly merged bounding boxes.
                                  - 'single_axle' contains remaining non-merged single axle boxes.
    :return: The updated annotation dictionary with modified 'annotated_regions' for each image.
    """
    for image in annotation["images"]:
        image_id = image["location"]
        updated_regions = []

        for region in image["annotated_regions"]: # Keep only non-axle annotations (car, other) in the updated dataset
            if "single_axle" not in region["tags"] and "grouped_axles" not in region["tags"]:
                updated_regions.append(region)


        if image_id in merged_boxes_by_image: # If this image has merged axle annotations, add them
            for merged_box in merged_boxes_by_image[image_id]["grouped_axles"]: # Add grouped_axles
                updated_regions.append({
                    "tags": ["grouped_axles"],
                    "region_type": "Box",
                    "region": {k: float(v) for k, v in merged_box.items()}  # Ensure float format
                })

            for single_box in merged_boxes_by_image[image_id]["single_axle"]: # Add single_axle boxes that were not merged
                updated_regions.append({
                    "tags": ["single_axle"],
                    "region_type": "Box",
                    "region": {k: float(v) for k, v in single_box.items()}
                })

        image["annotated_regions"] = updated_regions # Update the image's annotations with the corrected list

    return annotation



def merge_axle_trees(annotation):
    """
    Main function to process and merge axle trees.

    This function extracts single_axle bounding boxes, merges those that are
    close enough based on a predefined threshold, and updates the annotation
    dataset by replacing the merged boxes while keeping non-axle annotations intact.

    :param annotation: Dictionary containing the original JSON annotations.
                       It follows the format {'images': [{...}]}, where each image contains
                       'annotated_regions' with labeled bounding boxes.
    :return: Updated JSON annotation dictionary with merged grouped_axles and remaining single_axle boxes.
    """

    if "images" not in annotation: # Validate input: Ensure 'images' key exists in the annotation dictionary
        raise ValueError("🚨 The input JSON does not contain the 'images' key. Please check the annotation file.")

    single_axle_boxes_by_image = {} # Extract single_axle bounding boxes for each image
    for image in annotation["images"]:
        image_id = image["location"]
        single_axle_boxes_by_image[image_id] = [
            region["region"] for region in image["annotated_regions"] if "single_axle" in region["tags"]
        ]
    # Almacena todas las cajas de single_axle, organizadas por cada imagen.
    # La clave será el nombre del archivo de la imagen.
    # El valor será una lista con todas las coordenadas de single_axle de esa imagen.

    merged_boxes_by_image = merge_boxes(single_axle_boxes_by_image)

    updated_annotation = update_annotations(annotation, merged_boxes_by_image)

    return updated_annotation  # Return the final processed annotation dictionary





# original main
if __name__ == '__main__':
    # Load annotations from json file
    json_data = open_json_from_file('annotations.json')

    # Merge
    json_data = merge_axle_trees(json_data)
    pretty_print(json_data)

    # Saves new annotations to json file
    save_json_to_file(json_data, 'new_annotations.json')
