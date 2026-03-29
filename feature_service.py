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


def _ensure_model_ready() -> None:
	global _MODEL, _PREPROCESS

	if torch is None or models is None or transforms is None:
		raise RuntimeError(
			"Missing ML dependencies for feature service. "
			f"Import error: {_IMPORT_ERROR}"
		)

	if _MODEL is None:
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


def _embed_images(images: List[Image.Image]) -> List[List[float]]:
	_ensure_model_ready()

	with torch.no_grad():
		batch = torch.stack([_PREPROCESS(img) for img in images], dim=0)
		feats = _MODEL(batch).squeeze(-1).squeeze(-1)
		if feats.ndim == 1:
			feats = feats.unsqueeze(0)
		arr = feats.cpu().numpy().astype(np.float32)
		return arr.tolist()


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
		vectors = _embed_images([img])
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

		vectors = _embed_images(images)
		return {"features": vectors}
	except HTTPException:
		raise
	except Exception as exc:
		raise HTTPException(status_code=500, detail=f"Batch extraction failed: {exc}") from exc


if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=8001)
