import copy
import unittest
from test_case.test_replay_engine import World,page,step,recording

class TextBindingTests(unittest.TestCase):
    def action(self):
        source=page("City/country/region");source["elements"][0]["type"]="input"
        results=page("Nepal");final=page("Nepal","World clock")
        return recording("Search for Nepal and add it",[step(source,results,"City/country/region","text","Nepal"),step(results,final,"Nepal")])

    def test_current_task_replaces_text_and_dependent_city_target(self):
        engine=World([page("Home")]).engine();action=self.action();original=copy.deepcopy(action)
        bound=engine.bind_replay_text(action,"Go to World Clock App and Search for Abu Dhabi and add it")
        self.assertEqual(bound["element_sequence"][0]["action_params"]["text"],"Abu Dhabi")
        self.assertEqual(bound["element_sequence"][1]["target"]["content"],"Abu Dhabi")
        self.assertEqual(action,original)

    def test_guidance_overrides_current_task(self):
        engine=World([page("Home")]).engine();engine.user_guidance="Remove Nepal using backspace, write Abu Dhabi and replay the rest"
        bound=engine.bind_replay_text(self.action(),"Search for Nepal and add it")
        self.assertEqual(bound["element_sequence"][0]["action_params"]["text"],"Abu Dhabi")

    def test_no_explicit_value_keeps_recorded_text(self):
        engine=World([page("Home")],model=[{"parameters":[{"id":"0","status":"absent","confidence":.95}]}]).engine()
        self.assertEqual(engine.bind_replay_text(self.action(),"Add a city"),self.action())

    def test_search_and_add_extracts_city_only(self):
        engine=World([page("Home")]).engine()
        for task in ("Go to clock app, then world clock, search and add Abu Dhabi", "World clock search then add Abu Dhabi"):
            self.assertEqual(engine.explicit_text_value("City/country/region",task),"Abu Dhabi")

    def test_country_to_city_result_is_rebound_and_add_preserved(self):
        engine=World([page("Home")]).engine()
        source=page("City/country/region");results=page("Nepal","Kathmandu / Nepal");picker=page("Add")
        action=recording("Access World Clock App and Search for Nepal",[
            step(source,results,"City/country/region","text","Nepal"),
            step(results,picker,"Kathmandu / Nepal"),
            step(picker,page("Kathmandu","World clock"),"Add")])
        bound=engine.bind_replay_text(action,"World clock search and add Abu Dhabi")
        self.assertEqual(bound["element_sequence"][0]["action_params"]["text"],"Abu Dhabi")
        self.assertEqual(bound["element_sequence"][1]["target"]["content"],"Abu Dhabi")
        self.assertEqual(bound["element_sequence"][2]["target"]["content"],"Add")
        self.assertEqual(bound["final_screen"]["elements"][0]["content"],"Abu Dhabi")

    def test_query_dependent_result_is_rebound_for_non_clock_apps(self):
        engine=World([page("Home")]).engine()
        source=page("Search term");results=page("Nepal","Nepal details")
        action=recording("Search for Nepal and open it",[
            step(source,results,"Search term","text","Nepal"),
            step(results,page("Nepal details"),"Nepal details"),
        ])
        bound=engine.bind_replay_text(action,"Search for Abu Dhabi and open it")
        self.assertEqual(bound["element_sequence"][0]["action_params"]["text"],"Abu Dhabi")
        self.assertEqual(bound["element_sequence"][1]["target"]["content"],"Abu Dhabi")

    def test_city_completion_requires_destination_not_search_result(self):
        world=World([page("Abu Dhabi / UAE","Clear search field"),page("World clock","Abu Dhabi")]);engine=world.engine()
        engine.required_world_clock_city="Abu Dhabi"
        self.assertTrue(engine.pending_operations())
        world.position=1;engine.page=None
        self.assertEqual(engine.pending_operations(),[])
        world.screens[1]=page("World clock","Kathmandu");engine.page=None
        self.assertTrue(engine.pending_operations())

    def test_generic_model_extracts_missing_value_once_and_caches(self):
        world=World([page("Home")],model=[{"found":True,"value":"Alice Smith","confidence":.95,"source_quote":"Find Alice Smith"}]);engine=world.engine()
        for _ in range(2):
            self.assertEqual(engine.resolve_input_parameter("Name","Find Alice Smith"),"Alice Smith")
        self.assertEqual(len(world.calls),1)
        self.assertEqual(world.calls[0][0],"extract_input")
        self.assertFalse(world.calls[0][2])

    def test_ungrounded_extraction_is_rejected_and_not_retried(self):
        world=World([page("Home")],model=[{"found":True,"value":"Invented","confidence":1,"source_quote":"Find Alice"}]);engine=world.engine()
        for _ in range(2):
            self.assertIsNone(engine.resolve_input_parameter("Name","Find Alice"))
        self.assertEqual(len(world.calls),1)

    def test_fast_extraction_needs_no_model(self):
        world=World([page("Home")]);engine=world.engine()
        self.assertEqual(engine.resolve_input_parameter("Search","Search and add Abu Dhabi"),"Abu Dhabi")
        self.assertEqual(world.calls,[])

    def test_batch_parameter_extraction_has_no_verb_whitelist(self):
        response={"parameters":[{"id":"0","status":"explicit","value":"Abu Dhabi","source_quote":"add Abu Dhabi","confidence":.95}]}
        world=World([page("Home")],model=[response]);engine=world.engine()
        task="Go to world clock from Clock app, add Abu Dhabi"
        action=self.action()
        bound=engine.bind_replay_text(action,task)
        self.assertEqual(bound["element_sequence"][0]["action_params"]["text"],"Abu Dhabi")
        engine.bind_replay_text(bound,task)
        self.assertEqual(len(world.calls),1)
        self.assertEqual(world.calls[0][0],"extract_parameters")

    def test_multiple_fields_use_one_call(self):
        source=page("Name","Email")
        action=recording("Create contact",[step(source,source,"Name","text","Old"),step(source,source,"Email","text","old@example.com")])
        response={"parameters":[{"id":"0","status":"explicit","value":"Alice","source_quote":"Alice","confidence":.95},{"id":"1","status":"explicit","value":"a@example.com","source_quote":"a@example.com","confidence":.95}]}
        world=World([source],model=[response]);engine=world.engine()
        bound=engine.bind_replay_text(action,"Create contact Alice with a@example.com")
        self.assertEqual([s["action_params"]["text"] for s in bound["element_sequence"]],["Alice","a@example.com"])
        self.assertEqual(len(world.calls),1)

    def test_ambiguous_mapping_does_not_reuse_stored_default(self):
        world=World([page("Home")],model=[{"parameters":[{"id":"0","status":"ambiguous","confidence":.95}]}]);engine=world.engine()
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError,"Input parameter unresolved"):
                engine.bind_replay_text(self.action(),"Add that place")
        self.assertEqual(len(world.calls),1)

    def test_country_query_updates_separate_city_result(self):
        engine=World([page("Home")]).engine()
        source=page("City/country/region");results=page("Nepal","Kathmandu");picker=page("Add")
        action=recording("Search for Nepal",[step(source,results,"City/country/region","text","Nepal"),step(results,picker,"Kathmandu"),step(picker,page("Kathmandu"),"Add")])
        bound=engine.bind_replay_text(action,"Search for Abu Dhabi")
        self.assertEqual(bound["element_sequence"][1]["target"]["content"],"Abu Dhabi")
        self.assertTrue(bound["element_sequence"][1]["target"]["query_dependent"])
        self.assertEqual(bound["element_sequence"][2]["target"]["content"],"Add")

    def test_result_label_does_not_prove_input_is_filled(self):
        from replay_engine import input_value_present,target_index
        live=page("City/country/region","Abu Dhabi / UAE")
        live["elements"][0].update(type="input",editable=True)
        self.assertFalse(input_value_present("Abu Dhabi",live,live["elements"][0]))
        live["elements"][0]["content"]="Abu Dhabi"
        self.assertTrue(input_value_present("Abu Dhabi",live,live["elements"][0]))
        self.assertEqual(target_index({"content":"Abu Dhabi","query_dependent":True},live),1)

    def test_empty_parameter_response_has_one_bounded_recovery(self):
        recovered={"parameters":[{"id":"0","status":"explicit","value":"Sri Lanka",
                                  "source_quote":"add Sri Lanka","confidence":.95}]}
        world=World([page("Home")],model=[{},recovered]);engine=world.engine()
        bound=engine.bind_replay_text(self.action(),"Go to world clock and add Sri Lanka")
        self.assertEqual(bound["element_sequence"][0]["action_params"]["text"],"Sri Lanka")
        self.assertEqual([call[0] for call in world.calls],
                         ["extract_parameters","extract_parameters_recovery"])

    def test_guidance_resumes_same_matched_action_after_ambiguous_extraction(self):
        world=World([page("Home")],model=[{},
            {"parameters":[{"id":"0","status":"ambiguous","value":None,
                            "source_quote":"","confidence":.95}]}])
        engine=world.engine()
        requests=[]
        engine.assistance_callback=lambda payload: requests.append(payload) or {
            "info":"city/country/region=Sri Lanka"
        }
        bound=engine.bind_replay_text(self.action(),"Go to world clock and add that place")
        self.assertEqual(bound["action_id"],"stored")
        self.assertEqual(bound["element_sequence"][0]["action_params"]["text"],"Sri Lanka")
        self.assertEqual(len(requests),1)

    def test_unresolved_parameter_retains_plan_and_never_enters_full_react(self):
        action=self.action()
        plan={"action_id":"stored","relation":"equivalent","shared_prefix":1,
              "confidence":1.0,"reason":"same workflow with changed input"}
        ambiguous={"parameters":[{"id":"0","status":"ambiguous","value":None,
                                  "source_quote":"","confidence":.95}]}
        world=World([page("Home")],model=[plan,{},ambiguous],actions=[action])
        result=world.engine().run("Add that place")
        self.assertFalse(result["completed"])
        self.assertTrue(result["status"].startswith("uncertain: Input parameter unresolved"))
        self.assertEqual(result["plan"]["action_id"],"stored")
        self.assertEqual(result["react_steps"],0)
        self.assertEqual(world.commands,[])
