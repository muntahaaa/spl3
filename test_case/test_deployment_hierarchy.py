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
                self.assertEqual(calls[-1][-3],"rm")
                self.assertTrue(calls[-1][-1].startswith("/data/local/tmp/codex_deployment_"))
                self.assertEqual(calls[-1][-1],calls[0][-1])
            finally:
                os.chdir(old)

    def test_autocomplete_is_focused_input(self):
        xml='<hierarchy><node class="android.widget.AutoCompleteTextView" text="City/country/region" focused="true" bounds="[0,20][100,50]"/></hierarchy>'
        element=parse_hierarchy(xml,200,400)[0]
        self.assertEqual(element["type"],"input")
        self.assertTrue(element["editable"])
        self.assertTrue(element["focused"])

    def test_idle_failure_falls_back_without_cat_or_retry(self):
        calls=[];logs=[]
        def run(args,**kwargs):
            calls.append(args)
            return SimpleNamespace(returncode=0,stderr="",stdout="ERROR: could not get idle state.")
        with patch("deployment_hierarchy.subprocess.run",side_effect=run):
            self.assertEqual(capture_hierarchy("device","adb",{"width":200,"height":400},lambda _:None,logs.append),[])
        self.assertEqual(len(calls),2)
        self.assertFalse(any("cat" in args for args in calls))
        self.assertTrue(any("without retrying" in line for line in logs))

    def test_parent_and_child_tab_have_same_control_identity(self):
        xml='<hierarchy><node text="World clock" clickable="true" package="clock" bounds="[0,20][100,80]"><node text="World clock" bounds="[10,30][90,60]"/></node></hierarchy>'
        result=parse_hierarchy(xml,200,400)
        self.assertEqual(result[0]["control_id"],result[1]["control_id"])
