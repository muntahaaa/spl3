import json
import unittest
from typing import List,Dict,Any
from test_case.test_replay_integration import load_functions

class TaskMetadataTests(unittest.TestCase):
    def extract(self,chain):
        ns={"json":json,"List":List,"Dict":Dict,"Any":Any}
        load_functions("chain_evolve.py",{"extract_task_description","_source_step","_json_val"},ns)
        return ns["extract_task_description"](chain)

    def test_task_found_when_step_zero_is_not_first(self):
        chain=[{"source_page":{"other_info":{"step":2}}},{"source_page":{"other_info":json.dumps({"step":0,"task_info":{"description":"Open world clock"}})}}]
        self.assertEqual(self.extract(chain),"Open world clock")

    def test_target_page_metadata_is_supported(self):
        chain=[{"source_page":{},"target_page":{"other_info":{"task_info":{"description":"Start timer"}}}}]
        self.assertEqual(self.extract(chain),"Start timer")

    def test_missing_metadata_not_inferred_from_page_description(self):
        self.assertEqual(self.extract([{"source_page":{"description":"Alarm list"}}]),"Unknown task")
