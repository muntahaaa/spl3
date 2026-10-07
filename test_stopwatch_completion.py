import unittest
from test_replay_engine import World,page

class StopwatchCompletionTests(unittest.TestCase):
    def engine(self,labels=("0.00Seconds","Start","Stopwatch")):
        live=page(*labels)
        for element in live["elements"]:
            element["selected"] = element["content"] == "Stopwatch"
        world=World([live]);engine=world.engine()
        engine.required_operations=["start","reset"]
        return world,engine

    def test_zero_after_start_reset_completes_without_judge(self):
        world,engine=self.engine()
        engine.history=[{"status":"success","target":"start"},{"status":"success","target":"reset"}]
        self.assertTrue(engine.verify("Start Stopwatch then reset",True))
        self.assertEqual(world.calls,[])

    def test_initial_zero_without_execution_is_not_completion(self):
        world,engine=self.engine()
        self.assertFalse(engine.verify("Start Stopwatch then reset",True))
        self.assertEqual(world.calls,[])

    def test_nonzero_or_reset_before_start_cannot_confirm(self):
        world,engine=self.engine(("1.00Seconds","Start","Stopwatch"))
        engine.history=[{"status":"success","target":"start"},{"status":"success","target":"reset"}]
        self.assertFalse(engine.stopwatch_reset_complete("Start Stopwatch then reset"))
        world,engine=self.engine();engine.history=[{"status":"success","target":"reset"},{"status":"success","target":"start"}]
        self.assertFalse(engine.stopwatch_reset_complete("Start Stopwatch then reset"))

    def test_placeholder_action_never_taps(self):
        world=World([page("Start")],model=[{"action":"tap","element_id":0,"reason":"brief evidence"}])
        engine=world.engine()
        self.assertFalse(engine.react("Start Stopwatch then reset",None))
        self.assertEqual(world.commands,[])
        self.assertTrue(engine.failure.startswith("uncertain:"))
