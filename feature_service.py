"""
feature_service.py
==================
CPU-friendly FastAPI service that exposes:
  - POST /extract_single/?model_name=resnet50
  - POST /extract_batch/?model_name=resnet50

Response format:
	{"features": [[float, ...], ...]}
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from collections import OrderedDict
from threading import RLock
from io import BytesIO
from typing import List

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

try:
	import torch
	from torchvision import models, transforms
except Exception as exc:  # pragma: no cover - runtime dependency guard
	torch = None
	models = None
	transforms = None
	_IMPORT_ERROR = str(exc)
else:
	_IMPORT_ERROR = ""


app = FastAPI(title="ResNet50 Feature Service", version="1.0.0")

_MODEL = None
_PREPROCESS = None
_MODEL_LOCK = RLock()
_INFERENCE_GATE = asyncio.Semaphore(1)
_CACHE = OrderedDict()
_CACHE_SIZE = max(0, int(os.getenv("FEATURE_CACHE_SIZE", "512")))
_BATCH_SIZE = max(1, int(os.getenv("FEATURE_BATCH_SIZE", "8")))


def _initialize_model() -> None:
	global _MODEL, _PREPROCESS

	if torch is None or models is None or transforms is None:
		raise RuntimeError(
			"Missing ML dependencies for feature service. "
			f"Import error: {_IMPORT_ERROR}"
		)

	if _MODEL is None:
		# Bound CPU thread contention; configurable for this machine.
		torch.set_num_threads(max(1, int(os.getenv("FEATURE_CPU_THREADS", str(torch.get_num_threads())))))
		# Use torchvision defaults and strip classification head to get 2048-dim vectors.
		weights = models.ResNet50_Weights.DEFAULT
		base = models.resnet50(weights=weights)
		_MODEL = torch.nn.Sequential(*list(base.children())[:-1])
		_MODEL.eval()

		_PREPROCESS = transforms.Compose(
			[
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize(
					mean=[0.485, 0.456, 0.406],
					std=[0.229, 0.224, 0.225],
				),
			]
		)


def _bytes_to_rgb_image(data: bytes) -> Image.Image:
	try:
		return Image.open(BytesIO(data)).convert("RGB")
	except Exception as exc:
		raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc


def _ensure_model_ready() -> None:
    with _MODEL_LOCK:
        _initialize_model()


def _embed_images(images: List[Image.Image]) -> List[List[float]]:
    # Serialize model/cache access; concurrent callers cannot duplicate a cache miss.
    with _MODEL_LOCK:
        _ensure_model_ready()
        keys = [("resnet50-default-rgb-resize256-crop224-v1", img.size,
                 hashlib.sha256(img.tobytes()).digest()) for img in images]
        missing = OrderedDict()
        for key, image in zip(keys, images):
            if key not in _CACHE:
                missing.setdefault(key, image)
        computed = {}
        items = list(missing.items())
        with torch.inference_mode():
            for start in range(0, len(items), _BATCH_SIZE):
                chunk = items[start:start + _BATCH_SIZE]
                batch = torch.stack([_PREPROCESS(img) for _, img in chunk], dim=0)
                vectors = _MODEL(batch).flatten(1).cpu().numpy().astype(np.float32).tolist()
                computed.update((key, vector) for (key, _), vector in zip(chunk, vectors))
        result = []
        for key in keys:
            vector = computed[key] if key in computed else _CACHE[key]
            result.append(list(vector))
        for key, vector in zip(keys, result):
            if key in _CACHE:
                _CACHE.move_to_end(key)
            elif _CACHE_SIZE:
                _CACHE[key] = tuple(vector)
            while len(_CACHE) > _CACHE_SIZE:
                _CACHE.popitem(last=False)
        print(f"[FEATURE] images={len(images)}; unique cache misses={len(missing)}; batch limit={_BATCH_SIZE}; threads={torch.get_num_threads()}.")
        return result


@app.on_event("startup")
async def warm_up() -> None:
    started = time.monotonic()
    try:
        # Use the real preprocessing and model; no change to weights or vectors.
        await asyncio.to_thread(_embed_images, [Image.new("RGB", (224, 224))])
        print(f"[FEATURE] Model loaded and warmed in {time.monotonic()-started:.2f}s.")
    except Exception as exc:
        # Preserve the previous lazy-load retry behavior if startup loading fails.
        print(f"[FEATURE] Warm-up failed; extraction will retry loading: {exc}")


async def _embed_async(images):
    async with _INFERENCE_GATE:
        return await asyncio.to_thread(_embed_images, images)


@app.get("/health")
def health() -> dict:
	return {"status": "ok"}


@app.post("/extract_single/")
async def extract_single(model_name: str, file: UploadFile = File(...)) -> dict:
	if model_name.lower() != "resnet50":
		raise HTTPException(status_code=400, detail="Only model_name=resnet50 is supported")

	content = await file.read()
	if not content:
		raise HTTPException(status_code=400, detail="Uploaded file is empty")

	try:
		img = _bytes_to_rgb_image(content)
		vectors = await _embed_async([img])
		return {"features": vectors}
	except HTTPException:
		raise
	except Exception as exc:
		raise HTTPException(status_code=500, detail=f"Feature extraction failed: {exc}") from exc


@app.post("/extract_batch/")
async def extract_batch(model_name: str, files: List[UploadFile] = File(...)) -> dict:
	if model_name.lower() != "resnet50":
		raise HTTPException(status_code=400, detail="Only model_name=resnet50 is supported")
	if not files:
		raise HTTPException(status_code=400, detail="No files provided")

	try:
		images = []
		for fp in files:
			content = await fp.read()
			if not content:
				raise HTTPException(status_code=400, detail=f"Uploaded file is empty: {fp.filename}")
			images.append(_bytes_to_rgb_image(content))

		vectors = await _embed_async(images)
		return {"features": vectors}
	except HTTPException:
		raise
	except Exception as exc:
		raise HTTPException(status_code=500, detail=f"Batch extraction failed: {exc}") from exc


if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=8001)
