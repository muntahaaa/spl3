"""Offline behavioral tests: no Android device, database, or API required."""
import copy
from collections import deque
import unittest
from replay_engine import ReplayEngine, enrich_step, screen_score, target_index, values_present, hydrate_actions


def page(*labels):
    return {"screenshot": "test.png", "elements": [
        {"ID": i, "type": "text", "content": label, "bbox": [.1, .1+i*.12, .8, .18+i*.12]}
        for i, label in enumerate(labels)]}


def step(source, destination, label, atomic="tap", text=None):
    target = next(e for e in source["elements"] if e["content"] == label)
    return {"atomic_action": atomic, "action_params": {"text": text} if text else {},
            "target": target, "source": source, "destination": destination}


def recording(task, steps):
    return {"action_id": "stored", "source_task": task, "element_sequence": steps,
            "final_screen": steps[-1]["destination"]}


class World:
    def __init__(self, screens, model=None, actions=None, start=0, home_index=0):
        self.screens = screens
        self.position = start
        self.home_index = home_index
        self.calls = []
        self.commands = []
        self.home_count = 0
        self.actions = actions or []
        self.responses = deque(model or [])
        self.captures = 0

    def capture(self):
        self.captures += 1
        return copy.deepcopy(self.screens[self.position])

    def act(self, command):
        self.commands.append(command)
        self.position = min(self.position + 1, len(self.screens)-1)
        return True

    def home(self):
        self.home_count += 1
        self.position = self.home_index
        return True

    def model(self, kind, system, payload, vision):
        self.calls.append((kind, payload, vision))
        # Existing scenario scripts describe actions and terminal judgements.
        # Supply explicit incomplete progress for intermediate screens, leaving
        # the queued next action available for the following ReAct decision.
        if kind.startswith("judge") and self.responses and "action" in self.responses[0]:
            if self.position == len(self.screens)-1 and self.responses[0]["action"] == "done":
                self.responses.popleft()
            else:
                return {"complete": False, "confidence": .99, "evidence": "Intermediate screen", "missing": "Final goal"}
        return self.responses.popleft()

    def engine(self, max_steps=30):
        return ReplayEngine(self.capture, self.act, self.home, self.model, lambda: self.actions,
                            max_steps=max_steps, log=lambda _: None)


class ReplayTests(unittest.TestCase):
    def gallery(self):
        home = page("Home", "Gallery", "Clock")
        picture = page("Pictures", "Albums", "Camera photos")
        album = page("Albums", "Camera", "Screenshots")
        action = recording("Go to Gallery Albums", [step(home, picture, "Gallery"), step(picture, album, "Albums")])
        return [home, picture, album], action

    def test_case1_exact_replay_zero_model_calls(self):
        screens, action = self.gallery()
        world = World(screens, actions=[action])
        result = world.engine().run(action["source_task"])
        self.assertTrue(result["completed"], result)
        self.assertEqual(result["replayed_steps"], 2)
        self.assertEqual(world.calls, [])
        self.assertEqual(world.captures, 3)  # initial + each changed screen, no duplicates
        self.assertEqual(world.home_count, 0)

    def test_case2_resume_picture_without_home(self):
        screens, action = self.gallery()
        world = World(screens, actions=[action], start=1)
        result = world.engine().run(action["source_task"])
        self.assertTrue(result["completed"], result)
        self.assertEqual(result["resumed_step"], 2)
        self.assertEqual(result["replayed_steps"], 1)
        self.assertEqual(world.home_count, 0)
        self.assertEqual(world.calls, [])

    def test_case2_mismatch_home_then_start_first_step(self):
        screens, action = self.gallery()
        world = World(screens+[page("Unrelated", "Browser")], actions=[action], start=3)
        result = world.engine().run(action["source_task"])
        self.assertTrue(result["completed"], result)
        self.assertEqual(world.home_count, 1)
        self.assertEqual(result["resumed_step"], 1)
        self.assertEqual(world.calls, [])

    def test_case3_world_clock_replays_only_shared_prefix(self):
        home, clock, world_clock = page("Clock", "Home"), page("Alarms", "World clock", "Add"), page("World clock", "Dhaka", "London")
        old = recording("Set alarm at 10:30 pm", [step(home, clock, "Clock"), step(clock, page("Set alarm", "10:30 PM"), "Add")])
        model = [{"action_id": "stored", "relation": "related", "confidence": .95, "shared_prefix": 1},
                 {"action": "tap", "element_id": 1}, {"action": "done"},
                 {"complete": True, "confidence": .99, "evidence": "World clock displayed"}]
        world = World([home, clock, world_clock], model, [old])
        result = world.engine().run("Go to world clock")
        self.assertTrue(result["completed"], result)
        self.assertEqual(result["replayed_steps"], 1)
        self.assertEqual(result["react_steps"], 1)
        self.assertEqual([c[0] for c in world.calls], ["plan", "react", "judge_text"])

    def test_case3_changed_time_never_replays_old_value(self):
        home, clock = page("Clock", "Home"), page("Alarms", "Add")
        picker = page("Set alarm", "10:30 PM", "8:00 AM", "Save")
        old_set, new_set, saved = page("10:30 PM", "Save"), page("8:00 AM", "Save"), page("Alarms", "8:00 AM")
        old = recording("Set alarm at 10:30 pm", [step(home, clock, "Clock"), step(clock, picker, "Add"), step(picker, old_set, "10:30 PM"), step(old_set, page("Alarms", "10:30 PM"), "Save")])
        model = [{"action_id": "stored", "relation": "equivalent", "confidence": .99, "shared_prefix": 4},
                 {"action": "tap", "element_id": 2}, {"action": "tap", "element_id": 1}, {"action": "done"},
                 {"complete": True, "confidence": .99, "evidence": "8 AM alarm saved"}]
        world = World([home, clock, picker, new_set, saved], model, [old])
        result = world.engine().run("Set alarm at 8 am")
        self.assertEqual(result["plan"]["relation"], "related")
        self.assertEqual(result["replayed_steps"], 2)
        self.assertEqual(result["react_steps"], 2)
        self.assertEqual(world.commands[2]["bbox"], tuple(picker["elements"][2]["bbox"]))
        self.assertTrue(result["completed"], result)

    def test_case4_empty_catalog_skips_planner(self):
        world = World([page("World clock", "Dhaka")], [{"action": "done"}, {"complete": True, "confidence": .99, "evidence": "World clock visible"}])
        result = world.engine().run("Go to world clock")
        self.assertTrue(result["completed"], result)
        self.assertEqual(result["replayed_steps"], 0)
        self.assertNotIn("plan", result["llm_calls"])

    def test_case4_unrelated_catalog_falls_back(self):
        screens, action = self.gallery()
        world = World([page("World clock", "Dhaka")], [{"action_id": None, "relation": "none", "confidence": .99}, {"action": "done"}, {"complete": True, "confidence": .99, "evidence": "World clock visible"}], [action])
        self.assertTrue(world.engine().run("Go to world clock")["completed"])
        self.assertEqual(world.home_count, 0)

    def test_case5_recorded_inputs_zero_generation_calls(self):
        blank, typed, saved = page("Name", "Save"), page("Alice", "Save"), page("Alice", "Contact details")
        action = recording("Create contact Alice", [step(blank, typed, "Name", "text", "Alice"), step(typed, saved, "Save")])
        world = World([blank, typed, saved], actions=[action])
        result = world.engine().run("Create contact Alice")
        self.assertTrue(result["completed"], result)
        self.assertEqual(world.commands[0]["input_str"], "Alice")
        self.assertTrue(world.commands[0]["replace"])
        self.assertEqual(world.calls, [])

    def test_case5_new_form_one_batched_generation(self):
        blank, named, filled, saved = page("Name", "Phone", "Save"), page("Alex Example", "Phone", "Save"), page("Alex Example", "2025550143", "Save"), page("Alex Example", "2025550143", "Contact details")
        model = [{"values": {"name": "Alex Example", "phone": "2025550143"}},
                 {"action": "tap", "element_id": 2}, {"action": "done"},
                 {"complete": True, "confidence": .99, "evidence": "Contact saved"}]
        world = World([blank, named, filled, saved], model)
        result = world.engine().run("Create a contact")
        self.assertTrue(result["completed"], result)
        self.assertEqual(result["llm_calls"]["form"], 1)
        self.assertEqual([c["input_str"] for c in world.commands if c["action"] == "text"], ["Alex Example", "2025550143"])

    def test_case5_explicit_values_no_generation(self):
        blank, named, filled = page("Name", "Phone", "Save"), page("Bob", "Phone", "Save"), page("Bob", "2025550199", "Saved")
        world = World([blank, named, filled], [{"action": "done"}, {"complete": True, "confidence": .99, "evidence": "saved"}])
        result = world.engine().run("Create contact; name: Bob, phone: 2025550199")
        self.assertTrue(result["completed"], result)
        self.assertNotIn("form", result["llm_calls"])

    def test_malformed_planner_falls_back(self):
        screens, action = self.gallery()
        world = World([screens[-1]], [{"action_id": "stored", "confidence": "bad"}, {"action": "done"}, {"complete": True, "confidence": .99, "evidence": "Albums visible"}], [action])
        self.assertTrue(world.engine().run("Open albums please")["completed"])

    def test_no_stale_screen_after_capture_failure(self):
        screens, action = self.gallery()
        world = World(screens, actions=[action])
        world.capture = lambda: None
        result = world.engine().run(action["source_task"])
        self.assertFalse(result["completed"])
        self.assertEqual(world.commands, [])
        self.assertEqual(world.calls, [])

    def test_position_alone_does_not_match(self):
        live = page("Delete")
        target = page("Save")["elements"][0]
        self.assertIsNone(target_index(target, live))

    def test_duplicate_labels_reject_ambiguous_target(self):
        live = page("Save", "Save")
        live["elements"][1]["bbox"] = live["elements"][0]["bbox"]
        self.assertIsNone(target_index(live["elements"][0], live))

    def test_descriptive_add_target_matches_specific_live_add_control(self):
        target = {
            "content": "adding or creating a new item.",
            "type": "icon",
            "bbox": [.78, .06, .88, .11],
        }
        live = page("All alarms are off", "Add alarm", "More options")
        live["elements"][1].update({
            "clickable": True,
            "resource_id": "com.sec.android.app.clockpackage:id/menu_alarm_add",
        })
        self.assertEqual(target_index(target, live), 1)

    def test_descriptive_add_target_rejects_unrelated_controls(self):
        target = {"content": "creating something new", "type": "icon"}
        self.assertIsNone(target_index(target, page("More options", "Alarm")))

    def test_time_and_numeric_values_are_not_collapsed(self):
        self.assertFalse(values_present(["10:30 PM"], page("10:30 AM")))
        self.assertFalse(values_present(["8"], page("18")))
        self.assertEqual(screen_score(page("10:30 AM", "Save"), page("10:30 PM", "Save")), 0)

    def test_later_form_screen_cannot_skip_required_write(self):
        blank, typed, saved = page("Name", "Save"), page("Alice", "Save"), page("Alice", "Details")
        action = recording("Create Alice", [step(blank, typed, "Name", "text", "Alice"), step(typed, saved, "Save")])
        world = World([blank, typed, saved, page("Bob", "Save")], actions=[action], start=3)
        result = world.engine().run("Create Alice")
        self.assertEqual(world.home_count, 1)
        self.assertEqual(result["resumed_step"], 1)

    def test_empty_parser_uses_vision(self):
        world = World([page()], [{"action": "done"}, {"complete": True, "confidence": .99, "evidence": "visible screen"}])
        result = world.engine().run("Open Gallery")
        self.assertTrue(result["completed"], result)
        self.assertEqual(world.calls[0][0], "react_vision")
        self.assertEqual(world.calls[1][0], "judge_vision")
        self.assertIsNotNone(world.calls[0][2])

    def test_form_invalid_generated_email_is_not_typed(self):
        world = World([page("Name", "Email", "Save")], [{"values": {"name": "Alex", "email": "invalid"}}])
        result = world.engine().run("Create contact")
        self.assertFalse(result["completed"])
        self.assertEqual(world.commands, [])

    def test_failed_adb_never_reports_success(self):
        world = World([page("Gallery", "Home")], [{"action": "tap", "element_id": 0}])
        world.act = lambda _: False
        result = world.engine().run("Open gallery")
        self.assertFalse(result["completed"])
        self.assertEqual(result["status"], "action_error")

    def test_metrics_are_isolated(self):
        screens, action = self.gallery()
        first = World(screens, actions=[action]).engine().run(action["source_task"])
        second = World([page("Done")], [{"action": "done"}, {"complete": True, "confidence": .99, "evidence": "done"}]).engine().run("New task")
        self.assertEqual(first["llm_calls"], {})
        self.assertEqual(second["replayed_steps"], 0)

    def test_old_record_hydration(self):
        source, destination = page("Gallery"), page("Pictures", "Albums")
        old = {"action_id": "old", "element_sequence": [{"element_id": "e", "atomic_action": "tap"}]}
        metadata = {"e": [{"source": source, "destination": destination, "element": {"content": "Gallery", "bbox": source["elements"][0]["bbox"]}}]}
        hydrated = hydrate_actions([old], metadata)[0]
        self.assertEqual(hydrated["element_sequence"][0]["source"]["elements"], source["elements"])
        self.assertNotIn("source", old["element_sequence"][0])

    def test_home_failure_enters_react_without_replay(self):
        screens, action = self.gallery()
        world = World([page("Unexpected")], [{"action": "done"}, {"complete": False, "confidence": .99, "evidence": "not done"}]*2, [action])
        world.home = lambda: False
        result = world.engine(max_steps=2).run(action["source_task"])
        self.assertFalse(result["completed"])
        self.assertEqual(result["replayed_steps"], 0)

    def test_final_navigation_screen_is_checked_only_after_replay(self):
        screens, action = self.gallery()
        world = World(screens, actions=[action], start=2)
        result = world.engine().run(action["source_task"])
        self.assertTrue(result["completed"], result)
        self.assertEqual(len(world.commands), 1)
        self.assertEqual(world.calls, [])
        self.assertEqual(world.home_count, 0)
        self.assertEqual(result["metrics"]["replay_final_checks"], 1)

    def test_long_react_task_can_exceed_old_six_step_cap(self):
        screens = [page(f"Screen {i}", "Next") for i in range(13)]
        responses = [{"action": "tap", "element_id": 1} for _ in range(12)]
        responses += [{"action": "done"}, {"complete": True, "confidence": .99, "evidence": "Reached screen 12"}]
        world = World(screens, responses)
        result = world.engine(max_steps=20).run("Navigate through all screens")
        self.assertTrue(result["completed"], result)
        self.assertEqual(result["react_steps"], 12)

    def test_budget_exhaustion_is_not_success(self):
        world = World([page("One", "Next"), page("Two", "Next")], [{"action": "tap", "element_id": 1}, {"complete": False, "confidence": .99, "evidence": "not finished"}])
        result = world.engine(max_steps=1).run("Reach screen ten")
        self.assertFalse(result["completed"])
        self.assertEqual(result["status"], "budget_exhausted")

    def test_judge_vision_only_after_low_confidence(self):
        world = World([page("Some screen")], [{"action": "done"}, {"complete": False, "confidence": .2}, {"complete": True, "confidence": .99, "evidence": "visual proof"}])
        result = world.engine().run("Task")
        self.assertTrue(result["completed"], result)
        self.assertEqual([c[0] for c in world.calls], ["react", "judge_text", "judge_vision"])

    def test_note_form_generation_and_save(self):
        blank, title, body, saved = page("Title", "Note", "Save"), page("Shopping", "Note", "Save"), page("Shopping", "Buy milk", "Save"), page("Shopping", "Buy milk", "All notes")
        responses = [{"values": {"title": "Shopping", "note": "Buy milk"}}, {"action": "tap", "element_id": 2}, {"action": "done"}, {"complete": True, "confidence": .99, "evidence": "Note in list"}]
        world = World([blank, title, body, saved], responses)
        result = world.engine().run("Create a note")
        self.assertTrue(result["completed"], result)
        self.assertEqual(result["llm_calls"]["form"], 1)
        self.assertEqual(len(world.commands), 3)

    def test_recorded_form_values_reused_on_react_recovery(self):
        blank, typed = page("Name", "Phone", "Save"), page("Alice", "Phone", "Save")
        old = recording("Create contact", [step(blank, typed, "Name", "text", "Alice"), step(typed, page("Alice", "2025550100", "Save"), "Phone", "text", "2025550100")])
        world = World([blank], actions=[old])
        engine = world.engine()
        engine.resolve_form("Create contact", old)
        self.assertEqual(engine.form_values, {"name": "Alice", "phone": "2025550100"})
        self.assertEqual(world.calls, [])

    def test_catalog_selection_is_not_cached_across_tasks(self):
        screens, gallery = self.gallery()
        clock = recording("Open Clock", [step(page("Clock"), page("Alarms"), "Clock")])
        world = World(screens, actions=[gallery])
        engine = world.engine()
        self.assertEqual(engine.plan(gallery["source_task"])[0]["source_task"], gallery["source_task"])
        world.actions = [clock]
        self.assertEqual(engine.plan("Open Clock")[0]["source_task"], "Open Clock")

    def test_selected_tab_distinguishes_same_text(self):
        a, b = page("Pictures", "Albums"), page("Pictures", "Albums")
        a["elements"][0]["selected"] = True
        b["elements"][1]["selected"] = True
        self.assertLess(screen_score(a, b), .9)

    def test_executed_save_matching_stored_final_satisfies_threshold_policy(self):
        source = page("Save", "Contact")
        action = recording("Save contact", [step(source, source, "Save")])
        world = World([source], [{"complete": False, "confidence": .99, "evidence": "still editing"}, {"action": "done"}, {"complete": False, "confidence": .99, "evidence": "still editing"}], [action])
        result = world.engine(max_steps=2).run("Save contact")
        self.assertTrue(result["completed"])
        self.assertEqual(result["replayed_steps"],1)
        self.assertEqual(world.calls,[])

    def test_ambiguous_screens_do_not_jump_ahead(self):
        source = page("Pictures", "Albums")
        action = recording("Open album", [step(source, source, "Pictures"), step(source, page("Camera album"), "Albums")])
        world = World([source, source, page("Camera album")], actions=[action])
        engine = world.engine()
        self.assertIsNone(engine.align(action["element_sequence"], 2))

    def test_resume_form_only_when_previous_value_is_present(self):
        blank, typed, saved = page("Name", "Save"), page("Alice", "Save"), page("Alice", "Contact details")
        action = recording("Create Alice", [step(blank, typed, "Name", "text", "Alice"), step(typed, saved, "Save")])
        world = World([blank, typed, saved], actions=[action], start=1)
        result = world.engine().run("Create Alice")
        self.assertTrue(result["completed"], result)
        self.assertEqual(result["resumed_step"], 2)
        self.assertEqual(result["replayed_steps"], 1)
        self.assertEqual(world.home_count, 0)
        self.assertEqual(world.calls, [])

    def test_navigation_task_does_not_autofill_visible_fields(self):
        world = World([page("Name", "Phone", "Save")], [{"action": "done"}, {"complete": True, "confidence": .99, "evidence": "Contacts page visible"}])
        result = world.engine().run("Go to Contacts")
        self.assertTrue(result["completed"], result)
        self.assertNotIn("form", result["llm_calls"])
        self.assertEqual(world.commands, [])

    def test_fast_forward_to_further_stored_step(self):
        home = page("Home", "Clock")
        alarm = page("Alarm", "Timer")
        timer = page("02:00:00", "Start", "Timer")
        running = page("01:59:59", "Pause", "Timer")

        action = recording("Start timer", [
            step(home, alarm, "Clock"),
            step(alarm, timer, "Timer"),
            step(timer, running, "Start")
        ])

        # Phone opens Clock and is already on Timer tab (timer screen), then runs Step 3 (Start)
        world = World([home, timer, running], actions=[action])
        result = world.engine().run("Start timer")
        self.assertTrue(result["completed"], result)
        self.assertEqual(world.home_count, 0)


if __name__ == "__main__":
    unittest.main()
