import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from deployment_hierarchy import parse_hierarchy, capture_hierarchy

class HierarchyTests(unittest.TestCase):
    def test_parses_labels_search_selection_and_bounds(self):
        xml='<hierarchy><node text="Clock" bounds="[10,20][110,220]" selected="true" resource-id="launcher:clock"/><node class="android.widget.EditText" resource-id="launcher:search" bounds="[0,300][200,400]"/></hierarchy>'
        result=parse_hierarchy(xml,200,400)
        self.assertEqual(result[0]["bbox"],[.05,.05,.55,.55])
        self.assertTrue(result[0]["selected"])
        self.assertEqual(result[1]["content"],"Search")
        self.assertEqual(result[1]["type"],"input")

    def test_adb_failure_returns_empty_for_parser_fallback(self):
        logs=[]
        with patch("deployment_hierarchy.subprocess.run",side_effect=FileNotFoundError("adb")):
            self.assertEqual(capture_hierarchy("device","adb",{"width":200,"height":400},lambda _:None,logs.append),[])
        self.assertIn("falling back to OmniParser",logs[-1])

    def test_remote_fallback_removes_only_generated_file_and_tracks_xml(self):
        import os
        calls=[]; tracked=[]
        xml='<hierarchy><node text="Clock" bounds="[0,20][100,200]"/></hierarchy>'
        def run(args,**kwargs):
            calls.append(args)
            return SimpleNamespace(returncode=0,stderr="",stdout=xml if "cat" in args else "dump complete")
        old=os.getcwd()
        with tempfile.TemporaryDirectory() as folder,patch("deployment_hierarchy.subprocess.run",side_effect=run):
            try:
                os.chdir(folder)
                result=capture_hierarchy("device","adb",{"width":200,"height":400},tracked.append,lambda _:None)
                self.assertEqual(result[0]["content"],"Clock")
                self.assertEqual(len(tracked),1)
                self.assertEqual(calls[-1][-2],"rm")
                self.assertTrue(calls[-1][-1].startswith("/sdcard/codex_deployment_"))
                self.assertEqual(calls[-1][-1],calls[1][-1])
            finally:
                os.chdir(old)
