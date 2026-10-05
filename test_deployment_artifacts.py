import tempfile
from pathlib import Path
import unittest
from deployment_artifacts import DeploymentImages
from replay_engine import ReplayEngine, JUDGE_PROMPT


class DeploymentArtifactTests(unittest.TestCase):
    def image(self, root, name):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        return path

    def test_success_deletes_only_current_raw_and_parser_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = self.image(root, "log/screenshots/deployment/current.png")
            parser = self.image(root, "labeled_image/img/current.png")
            old = self.image(root, "log/screenshots/deployment/previous.png")
            stored = self.image(root, "log/screenshots/Gallery/recording.png")
            tracker = DeploymentImages(lambda _: None, root)
            tracker.track(raw)
            tracker.track(parser)
            result = tracker.finish(True)
            self.assertEqual(result["deleted"], 2)
            self.assertFalse(raw.exists())
            self.assertFalse(parser.exists())
            self.assertTrue(old.exists())
            self.assertTrue(stored.exists())

    def test_failure_preserves_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = self.image(root, "log/screenshots/deployment/current.png")
            tracker = DeploymentImages(lambda _: None, root)
            tracker.track(raw)
            self.assertEqual(tracker.finish(False)["deleted"], 0)
            self.assertTrue(raw.exists())

    def test_outside_directory_and_json_are_never_deleted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outside = self.image(root, "other/current.png")
            data = self.image(root, "log/screenshots/deployment/current.json")
            tracker = DeploymentImages(lambda _: None, root)
            tracker.track(outside)
            tracker.track(data)
            tracker.finish(True)
            self.assertTrue(outside.exists())
            self.assertTrue(data.exists())

    def test_late_parser_image_is_removed_after_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tracker = DeploymentImages(lambda _: None, root)
            tracker.finish(True)
            late = self.image(root, "labeled_image/img/late.png")
            tracker.track(late)
            self.assertFalse(late.exists())

    def test_exact_judge_system_prompt_is_logged(self):
        logs = []
        engine = ReplayEngine(lambda: None, lambda _: True, lambda: True,
                              lambda *args: {"complete": True}, lambda: [], log=logs.append)
        engine.ask("judge_vision", JUDGE_PROMPT, {"task": "test"}, vision=True)
        self.assertTrue(any("[JUDGE-SYSTEM-PROMPT]" in line and JUDGE_PROMPT in line for line in logs))
