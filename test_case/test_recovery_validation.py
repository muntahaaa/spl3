import unittest
from replay_engine import recovered_target_valid, target_index
from test_case.test_replay_engine import World,page,step,recording

class RecoveryValidationTests(unittest.TestCase):
    def candidate(self,label,typ="text"):
        return {"content":label,"type":typ,"bbox":[.1,.1,.5,.2]}

    def test_add_city_cannot_be_search_input_or_city_result(self):
        candidate=self.candidate("Add city")
        for target,atomic in (("search function.","tap"),("City/country/region","text"),("Kathmandu","tap")):
            self.assertFalse(recovered_target_valid({"content":target},candidate,atomic))

    def test_input_role_and_search_alias_are_supported(self):
        self.assertTrue(recovered_target_valid({"content":"search function."},self.candidate("Search"),"tap"))
        self.assertTrue(recovered_target_valid({"content":"City/country/region"},self.candidate("Search","input"),"text"))

    def test_unchanged_failed_text_cannot_reach_completion_judge(self):
        source=page("Search");source["elements"][0]["type"]="input"
        action=recording("Search for Nepal",[step(source,page("Nepal"),"Search","text","Nepal")])
        world=World([source,source]);engine=world.engine()
        engine.plan=lambda task:(action,"equivalent",1)
        result=engine.run("Search for Nepal")
        self.assertFalse(result["completed"])
        self.assertTrue(result["status"].startswith("uncertain:"))
        self.assertEqual(world.calls,[])
        self.assertEqual(result["history"][0]["status"],"unverified")

    def test_unknown_task_uses_actual_name_in_catalog_and_log(self):
        action=recording("Open Albums",[step(page("Albums"),page("Camera"),"Albums")])
        action["name"]="Open Albums";action["source_task"]="Unknown task"
        world=World([page("Albums")],actions=[action]);engine=world.engine();logs=[];engine.log=logs.append
        chosen,relation,limit=engine.plan("Open Albums")
        self.assertEqual(chosen["source_task"],"Open Albums")
        self.assertTrue(any("task='Open Albums'" in line for line in logs))

    def test_judge_cannot_override_unverified_replay(self):
        world=World([page("Add city")]);engine=world.engine();engine.replay_unverified=True
        self.assertFalse(engine.verify("Search for Nepal",True))
        self.assertEqual(world.calls,[])

    def test_search_description_resolves_without_model(self):
        self.assertEqual(target_index({"content":"search function."},page("Add city","Search")),1)
        self.assertIsNone(target_index({"content":"search function."},page("Search","Search")))

    def test_unchanged_focus_tap_proceeds_to_verified_text(self):
        source=page("City/country/region");source["elements"][0].update(type="input",editable=True,focused=True)
        destination=page("Nepal");destination["elements"][0].update(type="input",editable=True)
        stored_focus_destination=page("Keyboard","City/country/region")
        action=recording("Search for Nepal",[step(source,stored_focus_destination,"City/country/region"),step(source,destination,"City/country/region","text","Nepal")])
        world=World([source,source,destination]);engine=world.engine();engine.current_task="Search for Nepal"
        self.assertTrue(engine.replay(action,2))
        self.assertEqual([c["action"] for c in world.commands],["tap","text"])
        self.assertEqual(world.commands[1]["input_str"],"Nepal")
        self.assertEqual(world.calls,[])

    def test_duplicate_tab_nodes_share_one_control(self):
        live=page("World clock","World clock")
        for e in live["elements"]:
            e.update(package="clock",control_id="tab-world")
        self.assertIsNotNone(target_index({"content":"world clock"},live))
        live["elements"][1]["control_id"]="different-control"
        self.assertIsNone(target_index({"content":"world clock"},live))

    def test_matching_final_layout_cannot_skip_stored_text(self):
        world=World([page("World clock")]);engine=world.engine()
        engine.required_text_values=["Nepal"]
        engine.react_final_screen=page("World clock")
        self.assertFalse(engine.check_react_progress("Search for Nepal"))
        self.assertEqual(world.calls,[])
        self.assertEqual(engine.pending_operations(),["enter text: Nepal"])
        engine.history.append({"action":"text","params":{"input_str":"Nepal"},"status":"success"})
        self.assertEqual(engine.pending_operations(),[])

    def test_combined_result_label_resolves_and_replays_without_model(self):
        stored=page("Kathmandu"); live=page("Nepal","Kathmandu / Nepal")
        live["elements"][0].update(type="input",editable=True)
        live["elements"][1].update(clickable=True,package="clock")
        end=page("Add")
        action=recording("Search for Nepal",[step(stored,end,"Kathmandu")])
        world=World([live,end]);engine=world.engine();engine.start_from_home=True
        self.assertTrue(engine.replay(action,1))
        self.assertEqual(world.commands[0]["bbox"],tuple(live["elements"][1]["bbox"]))
        self.assertEqual(world.calls,[])
        self.assertTrue(recovered_target_valid({"content":"Kathmandu"},live["elements"][1],"tap"))
        self.assertFalse(recovered_target_valid({"content":"Kathmandu"},live["elements"][0],"tap"))

    def test_expanded_label_is_general_and_requires_unique_result(self):
        self.assertEqual(target_index({"content":"Abu Dhabi"},page("Abu Dhabi / UAE")),0)
        self.assertEqual(target_index({"content":"Alice Smith"},page("Alice Smith / Work")),0)
        self.assertIsNone(target_index({"content":"Kathmandu"},page("Kathmandu / Nepal","Kathmandu / Other")))
        self.assertIsNone(target_index({"content":"Kathmandu"},page("Kathmandux / Nepal")))
        self.assertIsNone(target_index({"content":"Delete"},page("Delete all")))
        live=page("Kathmandu / Nepal");live["elements"][0].update(type="input",editable=True)
        self.assertIsNone(target_index({"content":"Kathmandu"},live))

    def test_add_description_resolves_clickable_control_not_title(self):
        target={"content":"adding a new item or creating something new."}
        live=page("Add city","Add city","11:15:10")
        live["elements"][0].update(clickable=False,package="clock")
        live["elements"][1].update(clickable=True,package="clock")
        self.assertEqual(target_index(target,live),1)
        self.assertFalse(recovered_target_valid(target,live["elements"][0],"tap"))
        self.assertTrue(recovered_target_valid(target,live["elements"][1],"tap"))

    def test_add_role_is_general_and_rejects_ambiguity(self):
        target={"content":"Create something","role":"add"}
        self.assertEqual(target_index(target,page("New contact")),0)
        self.assertIsNone(target_index(target,page("Add city","New alarm")))
        live=page("Add city");live["elements"][0].update(enabled=False)
        self.assertIsNone(target_index(target,live))

    def test_add_description_replays_without_model(self):
        target_label="adding a new item or creating something new."
        source=page(target_label)
        action=recording("Add a city",[step(source,page("Search"),target_label)])
        live=page("Add city");live["elements"][0].update(clickable=True,package="clock")
        world=World([live,page("Search")]);engine=world.engine();engine.start_from_home=True
        self.assertTrue(engine.replay(action,1))
        self.assertEqual(world.calls,[])
        self.assertEqual(len(world.commands),1)

    def test_exact_description_outranks_add_role_alias(self):
        target={"content":"adding a new item or creating something new."}
        live=page("Add city","Adding a new item or creating something new.")
        # Parser-only elements have no authoritative clickability metadata.
        self.assertEqual(target_index(target,live),1)

    def test_exact_missing_step_asks_specific_question_only_after_bounded_recovery(self):
        source=page("Missing");destination=page("Next")
        action=recording("Exact stored task",[step(source,destination,"Missing")])
        live=page("Unrelated")
        world=World([live],model=[{"found":False,"element_id":None,"confidence":1,
                                  "reason":"Stored target is not visible"}])
        engine=world.engine();engine.start_from_home=True;engine.exact_task_match=True
        requests=[]
        engine.assistance_callback=lambda payload: requests.append(payload) or {"skip":True}
        self.assertFalse(engine.replay(action,1))
        self.assertEqual(len(requests),1)
        self.assertIn("What exact visible label or control",requests[0]["question"])
        self.assertIn("'missing'",requests[0]["question"])
        self.assertEqual([call[0] for call in world.calls],["recover_vision"])
