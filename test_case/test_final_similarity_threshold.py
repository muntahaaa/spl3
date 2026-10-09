import unittest
from unittest.mock import patch
from test_case.test_replay_engine import World,page,step,recording

class FinalSimilarityTests(unittest.TestCase):
    def replay_result(self,score):
        source,final=page("Albums"),page("Camera")
        action=recording("Open Albums",[step(source,final,"Albums")])
        world=World([source,final],actions=[action]);engine=world.engine()
        engine.plan=lambda task:(action,"equivalent",1)
        engine.replay=lambda action,limit:True
        engine.verify=lambda *args,**kwargs:False
        engine.react=lambda *args:False
        with patch("replay_engine.screen_score",return_value=score):
            return engine.run("Open Albums")

    def test_replay_success_above_point_seven(self):
        self.assertTrue(self.replay_result(.701)["completed"])

    def test_boundary_not_success(self):
        self.assertFalse(self.replay_result(.70)["completed"])

    def test_react_progress_uses_same_threshold(self):
        world=World([page("Camera")]);engine=world.engine();engine.react_final_screen=page("Camera")
        with patch("replay_engine.screen_score",return_value=.75):
            self.assertTrue(engine.check_react_progress("Open Albums"))
        self.assertEqual(world.calls,[])

    def test_pending_actions_still_block_success(self):
        world=World([page("Stopwatch")]);engine=world.engine();engine.react_final_screen=page("Stopwatch")
        engine.required_operations=["start","reset"]
        with patch("replay_engine.screen_score",return_value=.95):
            self.assertFalse(engine.check_react_progress("Start Stopwatch then reset"))
        self.assertEqual(world.calls,[])

class VisionSimilarityFallbackTests(unittest.TestCase):
    def test_low_similarity_compares_live_image_with_stored_content(self):
        world=World([page("Pictures","Camera")],model=[{"complete":True,"confidence":.95,"semantic_similarity":.9,"evidence":"Camera album visible on the requested Pictures page"}])
        engine=world.engine();engine.react_final_screen=page("Photos","Camera")
        with patch("replay_engine.screen_score",return_value=.6):
            self.assertTrue(engine.check_react_progress("Open Pictures"))
        kind,payload,vision=world.calls[0]
        import json
        payload=json.loads(payload)
        self.assertEqual(kind,"judge_vision")
        self.assertTrue(vision)
        self.assertIn("expected_final_elements",payload)
        self.assertEqual(payload["parsed_final_similarity"],.6)
        self.assertEqual(len(world.calls),1)

    def test_exact_boundary_uses_vision_and_does_not_assume_success(self):
        world=World([page("Pictures")],model=[{"complete":False,"confidence":.99,"semantic_similarity":.4,"missing":"Requested album is absent"}])
        engine=world.engine();engine.react_final_screen=page("Albums")
        with patch("replay_engine.screen_score",return_value=.7):
            self.assertFalse(engine.check_react_progress("Open Albums"))
        self.assertTrue(world.calls[0][2])
