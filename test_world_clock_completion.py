import unittest
from test_replay_engine import World,page

class WorldClockCompletionTests(unittest.TestCase):
    def test_dynamic_clock_content_confirms_navigation_without_model(self):
        world=World([page("World clock","Local time zone","5 hours behind","10.50","5.50")])
        engine=world.engine();engine.react_final_screen=page("World clock","9.00")
        self.assertTrue(engine.verify("Go to world clock",True))
        self.assertEqual(world.calls,[])

    def test_common_tab_labels_alone_do_not_confirm(self):
        world=World([page("Alarm","World clock","Stopwatch","Timer")])
        self.assertFalse(world.engine().world_clock_navigation_complete("Go to world clock"))

    def test_selected_tab_proves_destination(self):
        live=page("World clock","Alarm");live["elements"][0]["selected"]=True
        self.assertTrue(World([live]).engine().world_clock_navigation_complete("Open world clock"))

    def test_city_change_is_not_treated_as_navigation_only(self):
        live=page("World clock","Local time zone","5 hours behind","10.50")
        self.assertFalse(World([live]).engine().world_clock_navigation_complete("Add London to world clock"))

    def test_other_selected_tab_rejects_lookalike_content(self):
        live=page("World clock","Local time zone","5 hours behind","10.50","Alarm")
        live["elements"][-1]["selected"]=True
        self.assertFalse(World([live]).engine().world_clock_navigation_complete("Go to world clock"))
