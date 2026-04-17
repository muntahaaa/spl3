"""
tool/img_tool.py
================
Utilities for cropping UI-element images and calling the external
(CPU-based) ResNet50 feature-extraction service.
"""

import json
import os
import tempfile
from io import BytesIO
from typing import IO, List, Union, Dict

import requests
from PIL import Image
from langchain_core.tools import tool
from scipy.optimize import linear_sum_assignment
import config as config
import numpy as np


# ── feature extraction ────────────────────────────────────────────────────────

def extract_features(
    image_inputs: Union[str, List[str], IO, List[IO]],
    model_name: str,
) -> dict:
    """
    Call the external CPU-based embedding service.

    Accepts either a file path (str) or a file-like object (BytesIO / IO).
    Returns the JSON response from the service:
        {"features": [[float, ...]]}
    """
    is_single = not isinstance(image_inputs, list)
    inputs    = [image_inputs] if is_single else image_inputs

    url = (
        f"{config.Feature_URI}/extract_single/?model_name={model_name}"
        if is_single
        else f"{config.Feature_URI}/extract_batch/?model_name={model_name}"
    )

    key       = "file" if is_single else "files"
    files     = []
    tmp_paths = []

    try:
        for item in inputs:
            if isinstance(item, str):
                files.append((key, open(item, "rb")))
            else:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tmp_paths.append(tmp.name)
                item.seek(0)
                tmp.write(item.read())
                tmp.close()
                item.seek(0)
                files.append((key, open(tmp.name, "rb")))

        resp = requests.post(url, files=files, timeout=120)
        resp.raise_for_status()
        return resp.json()

    finally:
        for _, fp in files:
            try:
                fp.close()
            except Exception:
                pass
        for p in tmp_paths:
            try:
                os.unlink(p)
            except Exception:
                pass
# -- element similarity ────────────────────────────────────────────────────────────────
@tool
def element_similarity(
    page1: str,
    page2: str,
    json1: str,
    json2: str,
    feature_model: str = "resnet50",
    alpha: float = 0.7,
    beta: float = 0.3,
    distance_threshold: float = 0.5,
) -> Dict:
    """
    Calculate the similarity of elements in two pages.

    Parameters:
        page1: str, image path of the first page
        page2: str, image path of the second page
        json1: str, JSON file path of elements in the first page
        json2: str, JSON file path of elements in the second page
        feature_model: str, name of the feature extraction model
        alpha: float, weight of appearance features
        beta: float, weight of position features
        distance_threshold: float, matching threshold

    Returns:
        dict: Dictionary containing similarity scores and matching information
    """
    try:
        # 1. Load and preprocess data
        with open(json1, "r") as f:
            elements1 = json.load(f)
        with open(json2, "r") as f:
            elements2 = json.load(f)

        # 2. Crop and extract features
        features1 = _extract_element_features(page1, elements1, feature_model)
        features2 = _extract_element_features(page2, elements2, feature_model)

        # 3. Build distance matrix
        distance_matrix = _build_distance_matrix(
            features1, features2, elements1, elements2, alpha, beta
        )

        # 4. Use Hungarian algorithm for matching
        row_ind, col_ind = linear_sum_assignment(distance_matrix)

        # 5. Calculate final similarity and matching results
        matches = []
        total_similarity = 0
        valid_matches = 0

        for i, j in zip(row_ind, col_ind):
            distance = distance_matrix[i][j]
            if distance < distance_threshold:
                similarity = 1 - distance
                matches.append(
                    {
                        "element1": elements1[i],
                        "element2": elements2[j],
                        "similarity": float(similarity),
                    }
                )
                total_similarity += similarity
                valid_matches += 1

        # Calculate overall similarity
        overall_similarity = (
            total_similarity / valid_matches if valid_matches > 0 else 0.0
        )

        return {
            "similarity_score": float(overall_similarity),
            "matched_elements": valid_matches,
            "total_elements1": len(elements1),
            "total_elements2": len(elements2),
            "matches": matches,
            "status": "success",
            "message": "Successfully calculated element similarity",
        }

    except Exception as e:
        return {
            "similarity_score": 0.0,
            "status": "error",
            "message": f"Error occurred while calculating element similarity: {str(e)}",
        }


# ── element cropping ──────────────────────────────────────────────────────────

def element_img(page_path: str, elements_json: str, element_id: int) -> BytesIO:
    """
    Open the full-page screenshot and crop the bounding box of the
    element identified by `element_id`.

    Returns a PNG byte stream (BytesIO).
    """
    image        = Image.open(page_path)
    w, h         = image.size
    elements     = json.loads(elements_json)
  # Find the element with the specified ID
    target = next((e for e in elements if e.get("ID") == element_id), None)
    if target is None:
        raise ValueError(f"Element ID {element_id} not found in JSON")

    x1, y1, x2, y2 = target["bbox"]
    box = (
        max(0, int(x1 * w)),
        max(0, int(y1 * h)),
        min(w,  int(x2 * w)),
        min(h,  int(y2 * h)),
    )

    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"Invalid bbox for element {element_id}: {target['bbox']}")

    out = BytesIO()
    image.crop(box).save(out, format="PNG")
    out.seek(0)
    return out

def _extract_element_features(
    page_path: str, elements: List[Dict], feature_model: str
) -> List[np.ndarray]:
    """
    Crop elements from a page and extract features
    """
    try:
        # Load the original image
        image = Image.open(page_path)
        width, height = image.size
        # print(f"Processing image: {page_path}, size: {width}x{height}")

        features = []
        for i, element in enumerate(elements):
            try:
                # Get and normalize the bounding box
                bbox = element["bbox"]
                x1 = max(0, int(bbox[0] * width))
                y1 = max(0, int(bbox[1] * height))
                x2 = min(width, int(bbox[2] * width))
                y2 = min(height, int(bbox[3] * height))

                # Ensure a valid cropping area
                if x2 <= x1 or y2 <= y1:
                    print(f"Warning: Invalid bounding box for element {i}, skipping")
                    continue

                # Crop the element
                element_image = image.crop((x1, y1, x2, y2))

                # Ensure the cropped image is not empty
                if element_image.size[0] == 0 or element_image.size[1] == 0:
                    print(f"Warning: Cropped image for element {i} is empty, skipping")
                    continue

                # Create a temporary byte stream to store the cropped image
                img_byte_arr = BytesIO()
                element_image.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)

                # Extract features
                # print(f"Starting feature extraction for element {i}...")
                element_feature = extract_features(img_byte_arr, feature_model)
                # print(f"Feature extraction for element {i} succeeded")

                # Ensure the feature vector shape is correct
                feature_vector = np.array(element_feature["features"])
                if len(feature_vector.shape) > 1:
                    feature_vector = feature_vector.flatten()

                features.append(feature_vector)

            except Exception as e:
                print(f"Warning: Error processing element {i}: {str(e)}")
                print(f"Error type: {type(e)}")
                continue

        print(f"Successfully extracted features for {len(features)} elements")
        if not features:
            raise Exception("No element features were successfully extracted")

        return features

    except Exception as e:
        raise Exception(f"Error extracting element features: {str(e)}")


def _build_distance_matrix(
    features1: List[np.ndarray],
    features2: List[np.ndarray],
    elements1: List[Dict],
    elements2: List[Dict],
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Build a comprehensive distance matrix
    """
    n, m = len(features1), len(features2)
    distance_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            # Ensure the feature vector is one-dimensional
            f1 = features1[i].flatten()  # Flatten the feature vector
            f2 = features2[j].flatten()  # Flatten the feature vector

            # Calculate feature cosine distance
            norm1 = np.linalg.norm(f1)
            norm2 = np.linalg.norm(f2)

            if norm1 == 0 or norm2 == 0:
                feature_distance = (
                    1.0  # If the vector is a zero vector, set the maximum distance
                )
            else:
                # Calculate cosine similarity and convert to distance
                cosine_similarity = np.dot(f1, f2) / (norm1 * norm2)
                feature_distance = 1 - cosine_similarity

            # Calculate position distance
            pos_distance = _calculate_position_distance(
                elements1[i]["bbox"], elements2[j]["bbox"]
            )

            # Comprehensive distance
            distance_matrix[i][j] = alpha * feature_distance + beta * pos_distance

    return distance_matrix


def _calculate_position_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate the normalized position distance between two bounding boxes
    """
    # Calculate the center point
    center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
    center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]

    # Calculate the size
    size1 = [(bbox1[2] - bbox1[0]), (bbox1[3] - bbox1[1])]
    size2 = [(bbox2[2] - bbox2[0]), (bbox2[3] - bbox2[1])]

    # Calculate the center point distance
    center_distance = np.sqrt(
        (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
    )

    # Calculate the size difference
    size_difference = np.sqrt((size1[0] - size2[0]) ** 2 + (size1[1] - size2[1]) ** 2)

    # Normalize the total distance
    return (
        center_distance + size_difference
    ) / 4  # Divide by 4 because coordinates are in [0,1] range


if __name__ == "__main__":
    print(
        "img_tool module loaded. Use element_img(...) and extract_features(...) "
        "from project workflows or tests."
    )

