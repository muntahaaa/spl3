import copy
import unittest
from replay_engine import bounded_task_prefix, hydrate_actions, replay_source_match
from test_case.test_replay_engine import World, page, step, recording

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


class ProgressSafetyTests(unittest.TestCase):
    def test_identical_timer_final_screen_cannot_prove_unexecuted_operations(self):
        world=World([page("Timer","Start")])
        engine=world.engine()
        engine.required_operations=["start","pause","delete"]
        engine.react_final_screen=world.screens[0]
        self.assertFalse(engine.check_react_progress("Start timer, stop and delete"))
        self.assertEqual(world.calls,[])

    def test_ahead_timer_screen_skips_only_tab_navigation(self):
        home,alarm,timer,running,paused,final=page("Clock"),page("Alarm","Timer"),page("Timer","Start"),page("Timer","Pause"),page("Timer","Delete"),page("Timer","Start")
        action=recording("Start timer, stop it and delete",[step(home,alarm,"Clock"),step(alarm,timer,"Timer"),step(timer,running,"Start"),step(running,paused,"Pause"),step(paused,final,"Delete")])
        world=World([home,timer,running,paused,final],actions=[action])
        result=world.engine().run(action["source_task"])
        self.assertTrue(result["completed"],result)
        self.assertEqual([h.get("target") for h in result["history"]],["clock","start","pause","delete"])
        self.assertEqual(world.home_count,0)
        self.assertEqual(world.calls,[])

    def test_stopwatch_final_screen_requires_start_and_reset(self):
        world=World([page("Stopwatch","Start","Reset")])
        engine=world.engine()
        engine.required_operations=["start","reset"]
        engine.history=[{"status":"success","target":"stopwatch"}]
        self.assertFalse(engine.verify("Start Stopwatch then reset"))
        self.assertEqual(world.calls,[])


class CompletionEvidenceTests(unittest.TestCase):
    def test_world_clock_cannot_complete_on_alarm_editor(self):
        world=World([page("1, hour","00, minute","Save","Cancel")])
        engine=world.engine()
        self.assertFalse(engine.verify("Go to world clock",True))
        self.assertEqual(world.calls,[])

    def test_generic_judge_proof_is_rejected(self):
        world=World([page("World clock","London")],model=[{"complete":True,"confidence":.9,"evidence":"visible proof"}])
        self.assertFalse(world.engine().verify("Go to world clock",True))

    def test_another_selected_tab_is_not_world_clock(self):
        screen=page("Alarm","World clock","Timer")
        screen['elements'][0]['selected']=True
        engine=World([screen]).engine()
        self.assertFalse(engine.verify("Go to world clock"))


class BackRecoveryTests(unittest.TestCase):
    def test_missing_target_uses_back_before_retrying_stored_action(self):
        source,final=page("World clock"),page("London")
        action=recording("Go to world clock",[step(source,final,"World clock")])
        world=World([page("Cancel"),source,final])
        engine=world.engine()
        engine.start_from_home=True
        self.assertTrue(engine.replay(action,1))
        self.assertEqual([c['action'] for c in world.commands],['back','tap'])
        self.assertEqual(world.calls,[])


class CompletionLoopTests(unittest.TestCase):
    def test_identical_incomplete_evidence_is_judged_once(self):
        world=World([page("World clock","London")],model=[{"complete":False,"confidence":.99,"missing":"required city"}])
        engine=world.engine()
        self.assertFalse(engine.verify("Open London",True))
        self.assertFalse(engine.verify("Open London",True))
        self.assertEqual(len(world.calls),1)

    def test_done_loop_stops_without_repeated_capture_or_judge(self):
        world=World([page("World clock","London")],model=[{"action":"done"},{"complete":False,"confidence":.99,"missing":"not complete"},{"action":"done"}])
        result=world.engine().run("Open London",True)
        self.assertFalse(result['completed'])
        self.assertIn('completion_unconfirmed',result['status'])
        self.assertEqual(world.captures,1)
        self.assertEqual(sum(kind.startswith('judge') for kind,_,_ in world.calls),1)

class StalledFinalPageTests(unittest.TestCase):
    def test_matching_final_content_succeeds_without_model(self):
        world=World([page("World clock","London")])
        engine=world.engine(); engine.react_final_screen=page("World clock","London")
        self.assertTrue(engine.settle_unchanged_screen("Open world clock","unchanged"))
        self.assertEqual(world.calls,[])

    def test_semantic_match_uses_one_text_comparison(self):
        world=World([page("World clock","London")],model=[{"complete":True,"confidence":.95,"evidence":"London city entry visible on the World clock page; same requested result"}])
        engine=world.engine(); engine.react_final_screen=page("World clock","London city")
        self.assertTrue(engine.settle_unchanged_screen("Open world clock","unchanged"))
        self.assertEqual(len(world.calls),1)
        self.assertEqual(world.calls[0][0],"judge_stored_semantic")

    def test_pending_operations_prevent_false_success(self):
        world=World([page("Timer")]); engine=world.engine()
        engine.react_final_screen=page("Timer"); engine.required_operations=["start","stop","delete"]
        self.assertFalse(engine.settle_unchanged_screen("Start stop delete timer","unchanged"))
        self.assertTrue(engine.failure.startswith("uncertain:"))
        self.assertEqual(engine.stalled_state_report["pending_operations"],["start","stop","delete"])
        self.assertEqual(world.calls,[])

    def test_mismatch_reports_missing_content(self):
        world=World([page("Timer")],model=[{"complete":False,"confidence":.99,"missing":"London city entry absent"}])
        engine=world.engine(); engine.react_final_screen=page("World clock","London")
        self.assertFalse(engine.settle_unchanged_screen("Open London","unchanged"))
        self.assertIn("London city entry absent",engine.failure)
        self.assertIn("london",engine.stalled_state_report["differences"]["missing"])
