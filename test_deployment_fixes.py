import copy
import unittest
from replay_engine import bounded_task_prefix, hydrate_actions, replay_source_match
from test_replay_engine import World, page, step, recording

class DeploymentFixTests(unittest.TestCase):
    def timer_recording(self):
        screens = [page("Clock"),page("Timer"),page("Start"),page("Pause","Running"),page("Delete"),page("Timer")]
        steps = [step(screens[i],screens[i+1],label) for i,label in enumerate(("Clock","Timer","Start","Pause","Delete"))]
        return screens, recording("Start the timer, stop it and delete",steps)

    def test_timer_prefix_excludes_pause_delete_and_planner(self):
        screens, action = self.timer_recording()
        world = World(screens, actions=[action])
        chosen,relation,count = world.engine().plan("Start timer")
        self.assertEqual((relation,count),("prefix",3))
        self.assertEqual(world.calls,[])
        self.assertEqual(world.captures,0)

    def test_short_action_title_cannot_bypass_long_recorded_goal(self):
        screens, action = self.timer_recording()
        action["name"] = "Start timer"
        world = World(screens, actions=[action])
        _, relation, count = world.engine().plan("Start timer")
        self.assertEqual((relation, count), ("prefix", 3))

    def test_timer_prefix_cannot_reuse_recorded_duration_input(self):
        _,action = self.timer_recording()
        action["element_sequence"][1]["atomic_action"]="text"
        self.assertEqual(bounded_task_prefix("Start timer",action),0)

    def test_geometry_repairs_stale_label_without_mutating_recording(self):
        source, target = page("Clock"),page("Timer")
        action=recording("Open Clock",[step(source,target,"Clock")])
        action["element_sequence"][0]["target"]=copy.deepcopy(action["element_sequence"][0]["target"])
        action["element_sequence"][0]["target"]["content"]="0"
        repaired=hydrate_actions([action],{})[0]["element_sequence"][0]
        self.assertEqual(repaired["target"]["content"],"Clock")
        self.assertEqual(action["element_sequence"][0]["target"]["content"],"0")
        self.assertTrue(replay_source_match(repaired,source)[1])

    def test_recorded_status_target_is_not_guessed(self):
        source,target=page("Clock"),page("Timer")
        bad=step(source,target,"Clock")
        bad["target"]={"content":"0","bbox":[.1,.01,.2,.03],"type":"text"}
        repaired=hydrate_actions([recording("Open Clock",[bad])],{})[0]["element_sequence"][0]
        self.assertIn("target_metadata_error",repaired)
        self.assertFalse(replay_source_match(repaired,source)[1])

    def test_json_recovery_once_without_image(self):
        world=World([page("Clock")])
        engine=world.engine()
        failure=ValueError("no JSON")
        failure.response_text="Task is incomplete: still on home."
        calls=[]
        def model(kind,system,prompt,image):
            calls.append((kind,image))
            if len(calls)==1: raise failure
            return {"complete":False,"confidence":.9,"evidence":"home"}
        engine.model_fn=model
        engine.page=world.screens[0]
        self.assertFalse(engine.ask("judge_vision","schema",{},True)["complete"])
        self.assertEqual([kind for kind,_ in calls],["judge_vision","judge_vision_format"])
        self.assertIsNone(calls[1][1])
        self.assertEqual(engine.metrics["format_recoveries"],1)

    def test_unsuitable_react_target_never_reaches_adb(self):
        screen=page("9.46","Clock")
        screen["elements"][0]["bbox"]=[.05,.01,.12,.025]
        world=World([screen],model=[{"action":"tap","element_id":0},{"action":"tap","element_id":0}])
        result=world.engine().run("Start timer",True)
        self.assertFalse(result["completed"])
        self.assertEqual(world.commands,[])
        self.assertEqual([kind for kind,_,_ in world.calls],["react","react_vision"])

if __name__ == "__main__": unittest.main()
