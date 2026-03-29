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
from typing import IO, List, Union

import requests
from PIL import Image

import config as config


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
