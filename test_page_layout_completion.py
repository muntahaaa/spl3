import unittest
from replay_engine import page_layout_score,screen_score
from test_replay_engine import page,World

class PageLayoutTests(unittest.TestCase):
    def test_changed_time_and_am_pm_keep_same_layout(self):
        a,b=page("Alarm","10:30 AM","Save"),page("Alarm","8:15 PM","Save")
        self.assertGreater(page_layout_score(a,b),.7)
        self.assertEqual(screen_score(a,b),0)

    def test_selected_tab_mismatch_rejects_common_clock_layout(self):
        a,b=page("Alarm","Stopwatch","Timer"),page("Alarm","Stopwatch","Timer")
        a["elements"][0]["selected"]=True;b["elements"][1]["selected"]=True
        self.assertEqual(page_layout_score(a,b),0)

    def test_status_bar_values_are_ignored(self):
        a,b=page("Alarm","06:00"),page("Alarm","07:00")
        for screen,time in ((a,"10:00"),(b,"11:55")):
            screen["elements"].append({"content":time,"type":"text","bbox":[0,0,1,.04]})
        self.assertGreater(page_layout_score(a,b),.7)

    def test_input_values_remain_available_for_form_verification(self):
        a=page("Alarm","10:30");b=page("Alarm","08:15")
        page_layout_score(a,b)
        self.assertEqual(b["elements"][1]["content"],"08:15")
