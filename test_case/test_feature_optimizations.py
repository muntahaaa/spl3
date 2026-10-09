"""Feature API, upload ownership, ordering and cache contracts."""
import asyncio
from io import BytesIO
import tempfile
from pathlib import Path
from typing import Union, List, IO, Dict
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
import feature_service as service
from test_case.test_replay_integration import load_functions

class FeatureOptimizationTests(unittest.TestCase):
    def test_memory_upload_does_not_close_caller_stream(self):
        stream = BytesIO(b"image bytes")
        response = Mock()
        response.json.return_value = {"features": [[1.0]]}
        requests = SimpleNamespace(post=Mock(return_value=response))
        ns = {"requests": requests, "config": SimpleNamespace(Feature_URI="http://feature"), "Union": Union, "List": List, "IO": IO}
        load_functions("tool/img_tool.py", {"extract_features"}, ns)
        result = ns["extract_features"](stream, "resnet50")
        self.assertEqual(result, {"features": [[1.0]]})
        self.assertFalse(stream.closed)
        self.assertEqual(stream.tell(), 0)
        self.assertIs(requests.post.call_args.kwargs["files"][0][1][1], stream)

    def test_batch_crops_keep_order_and_skip_invalid_bbox(self):
        calls = []
        def extract(inputs, model):
            calls.append(inputs)
            return {"features": [[float(index)] for index in range(len(inputs))]}
        ns = {"Image": Image, "BytesIO": BytesIO, "np": np, "os": __import__("os"), "extract_features": extract, "List": List, "Dict": Dict}
        load_functions("tool/img_tool.py", {"_extract_element_features"}, ns)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder)/"screen.png")
            Image.new("RGB", (100,100)).save(path)
            elements = [{"bbox":[0,0,.2,.2]}, {"bbox":[.1,.1,.1,.1]}, {"bbox":[.5,.5,1,1]}]
            with patch("builtins.print"):
                vectors = ns["_extract_element_features"](path,elements,"resnet50")
        self.assertEqual(len(calls), 1)
        self.assertEqual([v.tolist() for v in vectors], [[0.0],[1.0]])

    def test_cache_deduplicates_orders_copies_and_evicts(self):
        import torch
        calls = []
        def model(batch):
            calls.append(len(batch))
            return batch.mean((2,3),keepdim=True)
        red = Image.new("RGB",(10,10),"red")
        blue = Image.new("RGB",(10,10),"blue")
        preprocess = lambda image: torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()
        with patch.object(service,"_ensure_model_ready"), patch.object(service,"_MODEL",model), patch.object(service,"_PREPROCESS",preprocess), patch.object(service,"_CACHE_SIZE",1), patch.object(service,"_BATCH_SIZE",8), patch("builtins.print"):
            service._CACHE.clear()
            vectors = service._embed_images([red,blue,red])
            self.assertEqual(calls,[2])
            self.assertEqual(vectors[0],vectors[2])
            vectors[0][0]=0
            again=service._embed_images([red])
            self.assertEqual(again[0][0],255)
            self.assertEqual(calls,[2])
            self.assertEqual(len(service._CACHE),1)
            mixed = service._embed_images([red,blue,red])
            self.assertEqual(mixed[0],mixed[2])
            self.assertEqual(calls,[2,1])
            service._CACHE.clear()

    def test_failed_batch_retains_individual_crop_recovery(self):
        calls = []
        def extract(inputs, model):
            calls.append(inputs)
            if isinstance(inputs, list):
                raise RuntimeError("batch unavailable")
            return {"features": [[7.0]]}
        ns = {"Image": Image, "BytesIO": BytesIO, "np": np, "os": __import__("os"), "extract_features": extract, "List": List, "Dict": Dict}
        load_functions("tool/img_tool.py", {"_extract_element_features"}, ns)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder)/"screen.png")
            Image.new("RGB", (100,100)).save(path)
            with patch("builtins.print"):
                vectors = ns["_extract_element_features"](path,[{"bbox":[0,0,1,1]}],"resnet50")
        self.assertEqual(len(calls), 2)
        self.assertEqual(vectors[0].tolist(), [7.0])

    def test_invalid_image_and_model_still_rejected(self):
        from fastapi import HTTPException, UploadFile
        with self.assertRaises(HTTPException) as caught:
            service._bytes_to_rgb_image(b"invalid")
        self.assertEqual(caught.exception.status_code,400)
        with self.assertRaises(HTTPException) as caught:
            asyncio.run(service.extract_single("other",UploadFile(file=BytesIO(b"image"))))
        self.assertEqual(caught.exception.status_code,400)

if __name__ == "__main__":
    unittest.main()
