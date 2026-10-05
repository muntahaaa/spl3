"""Contract tests for actual adapter/storage functions, with external I/O mocked.
AST loading avoids importing the app's eager network/dependency initialization.
The compiled function bodies are the real source, not test reimplementations.
"""
import ast
import asyncio
import io
import json
from pathlib import Path
import re
import shlex
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_replay_engine import World, page
import test_replay_engine as fixtures


def load_functions(path, names, namespace):
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    for node in selected:
        node.decorator_list = []
    futures = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "__future__"]
    exec(compile(ast.Module(body=futures + selected, type_ignores=[]), path, "exec"), namespace)
    return namespace


class IntegrationTests(unittest.TestCase):
    def test_default_run_task_routes_to_verified_engine(self):
        runner = Mock(return_value={"completed": True})
        ns = {"Dict": dict, "Any": object, "USE_PLAN_REUSE": True, "_run_verified_task": runner}
        load_functions("deployment.py", {"run_task"}, ns)
        self.assertTrue(ns["run_task"]("task", "device")["completed"])
        runner.assert_called_once_with("task", "device", 10, None, False)

    def test_real_deployment_adapter_exact_replay(self):
        screens, action = fixtures.ReplayTests().gallery()
        world = World(screens, actions=[action])
        tool = SimpleNamespace(invoke=lambda params: {"status": "success" if world.act(params) else "error"})
        bridge = Mock()
        ns = {"asyncio": asyncio, "json": json, "os": SimpleNamespace(path=SimpleNamespace(isfile=lambda _: True)),
              "create_execution_state": lambda _: {"current_step": 0},
              "take_screenshot": SimpleNamespace(invoke=lambda _: "fresh.png"),
              "omniparser_run": lambda _, **kwargs: "parsed.json", "_img_to_b64": lambda _: "image",
              "_device_size": lambda _: {"width": 1080, "height": 2400},
              "screen_action": tool, "_parse_action_result": lambda r: r["status"] == "success",
              "SCREENSHOT_SETTLE_SEC": 0, "press_home": lambda _: world.home(),
              "db": SimpleNamespace(get_all_high_level_actions=lambda **kwargs: [action], get_replay_metadata=lambda _: {}), "bridge": bridge}
        load_functions("deployment.py", {"_run_verified_task"}, ns)
        with patch("builtins.open", side_effect=lambda *a, **k: io.StringIO(json.dumps(world.capture()["elements"]))):
            result = ns["_run_verified_task"](action["source_task"], "device", log_callback=lambda _: None)
        self.assertTrue(result["completed"], result)
        self.assertEqual(result["replayed_steps"], 2)
        bridge.call_json.assert_not_called()
        self.assertEqual(world.commands[0]["device"], "device")
        self.assertIsInstance(world.commands[0]["x"], int)

    def test_recorded_sequence_hydrates_screens_and_input(self):
        from typing import Dict, Any, List, Tuple, Optional
        ns = {"Dict": Dict, "Any": Any, "List": List, "Tuple": Tuple, "Optional": Optional,
              "json": json, "ALLOWED_ATOMIC": {"text", "tap"}}
        load_functions("chain_evolve.py", {"_json_val", "_source_step", "normalise_params", "build_sequence_from_chain"}, ns)
        source = page("Name", "Save")
        source["other_info"] = json.dumps({"step": 0})
        destination = page("Alice", "Save")
        seq, reason = ns["build_sequence_from_chain"]([{"source_page": source, "target_page": destination,
            "element": {"element_id": "e", "action_type": "text", "parameters": json.dumps({"input_str": "Alice"}),
                        "other_info": json.dumps({"content": "Name", "type": "text"}), "bounding_box": json.dumps(source["elements"][0]["bbox"])}}])
        self.assertEqual(reason, "ok")
        self.assertEqual(seq[0]["action_params"]["text"], "Alice")
        # Current recordings store compact steps; deployment hydrates their
        # source/destination screens from the graph relationships.
        from replay_engine import hydrate_actions
        record = {"source": source, "destination": destination, "element": {
            "content": "Name", "bbox": source["elements"][0]["bbox"]}}
        hydrated = hydrate_actions([{"element_sequence": seq}], {"e": [record]})[0]["element_sequence"]
        self.assertEqual(hydrated[0]["destination"]["elements"], destination["elements"])
        self.assertEqual(hydrated[0]["role"], "input")

    def adb_function(self, subprocess_mock):
        ns = {"json": json, "re": re, "shlex": shlex, "subprocess": subprocess_mock,
              "_resolve_adb": lambda: "adb.exe"}
        load_functions("tool/adb_tools.py", {"screen_action"}, ns)
        return ns["screen_action"]

    def test_adb_text_focus_replace_and_shell_safe_punctuation(self):
        subprocess_mock = SimpleNamespace(run=Mock(return_value=SimpleNamespace(returncode=0)))
        func = self.adb_function(subprocess_mock)
        value = "Alice's note; $(test) & 50%s"
        result = json.loads(func(device="phone", action="text", x=10, y=20, input_str=value, replace=True))
        self.assertEqual(result["status"], "success")
        calls = subprocess_mock.run.call_args_list
        self.assertEqual(calls[0].args[0][-4:], ["input", "tap", "10", "20"])
        self.assertEqual(calls[1].args[0][-4:], ["input", "keycombination", "113", "29"])
        self.assertEqual(calls[2].args[0][-3:], ["input", "keyevent", "67"])
        self.assertTrue(all(isinstance(c.args[0], list) and not c.kwargs.get("shell") for c in calls))
        self.assertEqual(result["clicked_element"], {"x": 10, "y": 20})

    def test_unsupported_unicode_fails_without_typing(self):
        subprocess_mock = SimpleNamespace(run=Mock())
        result = json.loads(self.adb_function(subprocess_mock)(action="text", input_str="বাংলা"))
        self.assertEqual(result["status"], "error")
        subprocess_mock.run.assert_not_called()

    def test_actual_recorded_gallery_json_replays_and_resumes(self):
        # Uses the user's existing recording; no device or API access.
        root = Path("labeled_image/json_labeled_data")
        paths = [root/name for name in ["20261004_235028_015.json", "20261004_235104_056.json", "20261004_235135_529.json"]]
        if not all(p.exists() for p in paths):
            self.skipTest("Recorded Gallery JSON fixtures are not present")
        from test_replay_engine import recording
        screens = [{"screenshot": "recorded.png", "elements": json.loads(p.read_text())} for p in paths]
        steps = []
        for i, target_id in enumerate((25, 21)):
            target = next(e for e in screens[i]["elements"] if e["ID"] == target_id)
            steps.append({"atomic_action": "tap", "target": target, "source": screens[i], "destination": screens[i+1]})
        action = recording("Go to gallery and then navigate to the Album", steps)
        for start in (0, 1):
            with self.subTest(start=start):
                world = World(screens, actions=[action], start=start)
                result = world.engine().run(action["source_task"])
                self.assertTrue(result["completed"], result)
                self.assertEqual(result["replayed_steps"], 2-start)
                self.assertEqual(world.calls, [])
                self.assertEqual(world.home_count, 0)


if __name__ == "__main__":
    unittest.main()
