import unittest
from test_replay_engine import World,page

class TabRecoveryTests(unittest.TestCase):
    def test_back_finds_tab_without_home_or_swipe(self):
        world=World([page("Timer editor"),page("Alarm")]);engine=world.engine()
        self.assertEqual(engine.recover_missing_tab({"content":"Alarm"},"Clock"),0)
        self.assertEqual([c["action"] for c in world.commands],["back"])
        self.assertEqual(world.home_count,0)

    def test_home_icon_precedes_launcher_search(self):
        world=World([page("Missing"),page("Still missing"),page("Clock"),page("Alarm")])
        engine=world.engine();events=[]
        original_act=engine.act
        engine.act=lambda command,mode:events.append(command["action"]) or original_act(command,mode)
        def home():
            events.append("home");world.position=2;return True
        engine.home_fn=home
        self.assertEqual(engine.recover_missing_tab({"content":"Alarm"},"Clock"),0)
        self.assertEqual(events,["back","home","tap"])

    def test_last_resort_swipes_searches_types_and_opens_app(self):
        field=page("Search");field["elements"][0]["type"]="input"
        world=World([page("Missing"),page("Still missing"),page("Home"),page("Search"),field,page("Clock"),page("Alarm")])
        engine=world.engine();events=[];original_act=engine.act
        engine.act=lambda command,mode:events.append(command["action"]) or original_act(command,mode)
        def home():
            events.append("home");world.position=2;return True
        engine.home_fn=home
        self.assertEqual(engine.recover_missing_tab({"content":"Alarm"},"Clock"),0)
        self.assertEqual(events,["back","home","swipe","tap","text","tap"])
        self.assertEqual(world.commands[-2]["input_str"],"Clock")
        self.assertEqual(world.calls,[])
