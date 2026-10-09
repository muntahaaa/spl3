import unittest
from test_case.test_replay_engine import World,page,step,recording

class VisualRecoveryTests(unittest.TestCase):
    def test_changed_label_located_visually_without_home(self):
        source=page("Stopwatch"); live=page("Elapsed time"); final=page("Start")
        action=recording("Open Stopwatch",[step(source,final,"Stopwatch")])
        world=World([live,live,final],model=[{"found":True,"element_id":0,"confidence":.95,"reason":"Elapsed time control is the Stopwatch tab"}])
        engine=world.engine();engine.start_from_home=True
        self.assertTrue(engine.replay(action,1))
        self.assertEqual(world.home_count,0)
        self.assertEqual([c["action"] for c in world.commands],["back","tap"])
        self.assertEqual(world.calls[0][0],"recover_vision")

    def test_failed_visual_lookup_requests_guidance_without_action(self):
        source=page("Stopwatch");action=recording("Open Stopwatch",[step(source,page("Start"),"Stopwatch")])
        world=World([page("Alarm")],model=[{"found":False,"confidence":.99,"reason":"Stopwatch tab not visible"}])
        engine=world.engine();engine.start_from_home=True; requests=[]
        engine.assistance_callback=lambda payload:requests.append(payload) or {"skip":True}
        self.assertFalse(engine.replay(action,1))
        self.assertEqual(len(requests),1)
        self.assertEqual(world.commands,[{"action":"back"}])
        self.assertEqual(world.home_count,0)

    def test_template_proof_cannot_complete_task(self):
        world=World([page("Stopwatch")],model=[{"complete":True,"confidence":.9,"evidence":"visible proof"}])
        self.assertFalse(world.engine().verify("Open Stopwatch",True))

    def test_pending_reset_blocks_judge_even_on_final_page(self):
        world=World([page("Stopwatch","Start")]);engine=world.engine();engine.required_operations=["start","reset"]
        self.assertFalse(engine.verify("Start Stopwatch then reset",True))
        self.assertEqual(world.calls,[])
