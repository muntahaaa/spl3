import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from deployment_cases import retry_case_cleanup

class CleanupRetryTests(unittest.TestCase):
    def test_retry_cleans_vectors_and_reports_success(self):
        kinds=SimpleNamespace(**{name:SimpleNamespace(value=name.lower()) for name in ("PAGE","ELEMENT","ACTION")})
        calls=[];vector=SimpleNamespace(delete_vectors=lambda ids,kind:calls.append((ids,kind.value)) or True)
        old=os.getcwd()
        with tempfile.TemporaryDirectory() as folder,patch.dict("sys.modules",{"data.vector_db":SimpleNamespace(NodeType=kinds)}):
            try:
                os.chdir(folder);path=Path("log/case_deletions/retry.json");path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"graph_deleted":True,"pages":["p"],"elements":["e"],"actions":["a"]}))
                result=retry_case_cleanup(vector,str(path))
                self.assertEqual(result["status"],"deleted")
                self.assertEqual(len(calls),3)
            finally: os.chdir(old)

    def test_outside_manifest_is_rejected(self):
        with self.assertRaises(ValueError):retry_case_cleanup(None,"outside.json")
