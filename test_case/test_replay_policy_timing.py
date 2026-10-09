"""Regression tests for final-only replay and observable, bounded remote waits."""
import ast
import asyncio
import os
from pathlib import Path
import subprocess
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import replay_engine
from replay_engine import ReplayEngine, screen_score
from deployment_progress import run_with_progress
from test_case.test_replay_engine import World, page, step, recording
from test_case.test_replay_integration import load_functions


class ReplayPolicyTests(unittest.TestCase):
    def gallery(self):
        home = page('Home', 'Gallery', 'Clock')
        picture = page('Pictures', 'Albums', 'Camera photos')
        album = page('Albums', 'Camera', 'Screenshots')
        action = recording('Go to Gallery Albums', [step(home, picture, 'Gallery'), step(picture, album, 'Albums')])
        return [home, picture, album], action

    def test_equivalent_gallery_album_wording_matches_before_capture(self):
        screens, action = self.gallery()
        action['source_task'] = 'Open Gallery app and switch to Albums tab'
        world = World(screens, actions=[action])
        events = []
        def load():
            events.append('catalog')
            self.assertEqual(world.captures, 0)
            return world.actions
        def capture():
            events.append('capture')
            return world.capture()
        engine = ReplayEngine(capture, world.act, world.home, world.model, load, log=lambda _: None)
        result = engine.run('Go to gallery and switch to Album tab')
        self.assertTrue(result['completed'], result)
        self.assertEqual(events[0], 'catalog')
        self.assertEqual(result['replayed_steps'], 2)
        self.assertEqual(world.calls, [])
        self.assertEqual(result['metrics']['local_semantic_matches'], 1)

    def test_exact_and_prefix_plans_do_not_capture(self):
        screens, action = self.gallery()
        for task in (action['source_task'], 'Open Gallery'):
            world = World(screens, actions=[action])
            engine = world.engine()
            chosen, relation, count = engine.plan(task)
            self.assertIsNotNone(chosen)
            self.assertEqual(world.captures, 0)
            self.assertEqual(world.calls, [])

    def test_general_semantic_planner_runs_before_capture(self):
        screens, action = self.gallery()
        world = World(screens, actions=[action])
        events = []
        def model(kind, system, payload, vision):
            events.append(kind)
            self.assertEqual(world.captures, 0)
            self.assertIsNone(vision)
            return {'action_id': 'stored', 'relation': 'equivalent', 'confidence': .99}
        def capture():
            events.append('capture')
            return world.capture()
        result = ReplayEngine(capture, world.act, world.home, model, lambda: world.actions, log=lambda _: None).run('Please display my Gallery albums')
        self.assertTrue(result['completed'], result)
        self.assertEqual(events[0], 'plan')

    def test_local_navigation_matching_does_not_confuse_other_goals(self):
        _, action = self.gallery()
        action['source_task'] = 'Open Gallery app and switch to Albums tab'
        self.assertTrue(replay_engine.equivalent_navigation('Go to gallery and navigate to the Album', action))
        for task in ('Go to gallery', 'Go to gallery and delete Album', 'Go to gallery and switch to Pictures tab', 'Set alarm at 8 am'):
            self.assertFalse(replay_engine.equivalent_navigation(task, action), task)
        action['element_sequence'] = []
        self.assertFalse(replay_engine.equivalent_navigation('Go to gallery and switch to Album tab', action))

    def test_world_clock_intent_matches_despite_generic_recorded_target(self):
        source, final = page('Clock', 'World clock'), page('World clock', 'Dhaka')
        action = recording('Open world clock', [step(source, final, 'World clock')])
        action['element_sequence'][0]['target'] = dict(action['element_sequence'][0]['target'], content='navigation or navigation functionality.')
        world = World([source, final], actions=[action])
        engine = world.engine()
        chosen, relation, count = engine.plan('Go to world clock')
        self.assertEqual(chosen['action_id'], 'stored')
        self.assertEqual(relation, 'equivalent')
        self.assertEqual(count, 1)
        self.assertEqual(world.calls, [])
        self.assertEqual(world.captures, 0)
        self.assertTrue(replay_engine.replay_source_match(chosen['element_sequence'][0], source)[1])
        self.assertEqual(chosen['element_sequence'][0]['target_repair'], 'unique recorded source bbox')

    def test_gallery_open_request_reuses_only_first_step_without_models(self):
        screens, action = self.gallery()
        action['source_task'] = 'Go to gallery and switch to Album tab'
        world = World(screens, actions=[action])
        result = world.engine().run('Go to Gallery app')
        self.assertTrue(result['completed'], result)
        self.assertEqual(result['plan']['relation'], 'prefix')
        self.assertEqual(result['replayed_steps'], 1)
        self.assertEqual(result['react_steps'], 0)
        self.assertEqual(world.position, 1)
        self.assertEqual(world.calls, [])
        self.assertEqual(result['metrics']['replay_final_checks'], 1)

    def test_already_open_gallery_prefix_finishes_without_tapping_albums(self):
        screens, action = self.gallery()
        world = World(screens, actions=[action], start=1)
        result = world.engine().run('Open Gallery')
        self.assertTrue(result['completed'], result)
        self.assertEqual(world.commands, [])
        self.assertEqual(world.calls, [])
        self.assertEqual(world.home_count, 0)

    def test_simple_app_open_synonyms_and_compound_goal_rejection(self):
        _, action = self.gallery()
        for task in ('Go to Gallery app', 'Please open the Gallery app.', 'Launch Gallery', 'Navigate to Gallery'):
            self.assertEqual(replay_engine.app_open_prefix(task, action), 1, task)
        for task in ('Go to Gallery and delete a photo', 'Go to Gallery Albums', 'Open Gallery with album tab', 'Go to Clock app'):
            self.assertEqual(replay_engine.app_open_prefix(task, action), 0, task)

    def test_prefix_requires_recorded_target_and_destination(self):
        _, action = self.gallery()
        action['element_sequence'][0]['target']['content'] = 'Clock'
        self.assertEqual(replay_engine.app_open_prefix('Open Gallery', action), 0)
        _, action = self.gallery()
        action['element_sequence'][0]['destination'] = {}
        self.assertEqual(replay_engine.app_open_prefix('Open Gallery', action), 0)

    def test_prefix_does_not_cross_input_or_commit(self):
        screens, action = self.gallery()
        action['element_sequence'].insert(0, step(screens[0], screens[0], 'Home', atomic='text', text='old value'))
        self.assertEqual(replay_engine.app_open_prefix('Open Gallery', action), 0)

    def test_conflicting_prefix_recordings_require_planner(self):
        screens, action = self.gallery()
        import copy
        other = copy.deepcopy(action)
        other['action_id'] = 'different'
        other['element_sequence'][0]['destination'] = page('Unrelated screen')
        world = World(screens, actions=[action, other], model=[{'relation': 'none', 'confidence': .99}])
        engine = world.engine()
        engine.observe()
        self.assertEqual(engine.plan('Open Gallery'), (None, 'none', 0))
        self.assertEqual([c[0] for c in world.calls], ['plan'])

    def test_user_logged_live_home_matches_recorded_gallery_prefix(self):
        import json
        root = Path('labeled_image/json_labeled_data')
        names = ['20261005_125018_794.json', '20261004_235028_015.json', '20261004_235104_056.json', '20261004_235135_529.json']
        if not all((root/name).exists() for name in names):
            self.skipTest('User live/recorded JSON fixtures unavailable')
        live, home, picture, album = [dict(screenshot='fixture.png', elements=json.loads((root/name).read_text())) for name in names]
        action = recording('Go to gallery and then navigate to the Album', [step(home, picture, 'Gallery'), step(picture, album, 'Albums')])
        world = World([live, picture, album], actions=[action])
        result = world.engine().run('Go to Gallery app')
        self.assertTrue(result['completed'], result)
        self.assertEqual(result['replayed_steps'], 1)
        self.assertEqual(world.calls, [])
        self.assertEqual(world.home_count, 0)

    def test_visible_gallery_target_proceeds_despite_major_layout_drift(self):
        screens, action = self.gallery()
        screens[0] = page('Gallery', 'Changed wallpaper', 'New widget 42')
        world = World(screens, actions=[action])
        result = world.engine().run('Go to Gallery')
        self.assertTrue(result['completed'], result)
        self.assertEqual(result['replayed_steps'], 1)
        self.assertEqual(world.home_count, 0)
        self.assertEqual(world.calls, [])

    def test_after_home_unique_gallery_target_allows_replay(self):
        screens, action = self.gallery()
        changed_home = page('Gallery', 'New widget 42')
        world = World([page('Unrelated app'), changed_home, screens[1]], actions=[action], home_index=1)
        result = world.engine().run('Go to Gallery')
        self.assertTrue(result['completed'], result)
        self.assertEqual(world.home_count, 1)
        self.assertEqual(result['replayed_steps'], 1)
        self.assertEqual(world.calls, [])

    def test_source_matching_rejects_duplicate_target_and_wrong_app(self):
        screens, action = self.gallery()
        duplicate = page('Gallery', 'Gallery')
        duplicate['elements'][1]['bbox'] = duplicate['elements'][0]['bbox']
        self.assertFalse(replay_engine.replay_source_match(action['element_sequence'][0], duplicate)[1])
        action['element_sequence'][0]['source']['app_package'] = 'expected.app'
        live = page('Gallery')
        live['app_package'] = 'other.app'
        self.assertFalse(replay_engine.replay_source_match(action['element_sequence'][0], live)[1])

    def test_replay_ignores_intermediate_screen_drift_and_checks_final_once(self):
        screens, action = self.gallery()
        # An unrelated changed counter would previously fail the whole-page match.
        screens[1] = page('Pictures', 'Albums', 'New photos 42')
        world = World(screens, actions=[action])
        logs = []
        engine = ReplayEngine(world.capture, world.act, world.home, world.model, lambda: world.actions, log=logs.append)
        with patch.object(replay_engine, 'screen_score', wraps=screen_score) as compare:
            result = engine.run(action['source_task'])
        self.assertTrue(result['completed'], result)
        self.assertEqual(result['replayed_steps'], 2)
        self.assertEqual(result['metrics']['replay_final_checks'], 1)
        # Engine deep-copies the action; check values for the final comparison.
        final_checks = [c for c in compare.call_args_list if c.args[0] == action['final_screen'] and c.args[1] == screens[-1]]
        self.assertEqual(len(final_checks), 1)
        self.assertEqual(world.calls, [])
        self.assertNotIn('destination similarity', '\n'.join(logs))

    def test_semantic_equivalent_replays_without_per_step_judges(self):
        screens, action = self.gallery()
        world = World(screens, [{'action_id': 'stored', 'relation': 'equivalent', 'confidence': .95}], [action])
        result = world.engine().run('Please go to Gallery Albums')
        self.assertTrue(result['completed'], result)
        self.assertEqual([c[0] for c in world.calls], ['plan'])
        self.assertEqual(result['metrics']['replay_final_checks'], 1)

    def test_react_checks_completion_after_every_action(self):
        screens = [page('Start', 'Next'), page('Intermediate', 'Next'), page('Requested final screen')]
        world = World(screens)
        calls = []
        def model(kind, system, payload, vision):
            calls.append((kind, world.position))
            if kind == 'react':
                return {'action': 'tap', 'element_id': 1}
            self.assertEqual(kind, 'judge_text')
            return {'complete': world.position == 2, 'confidence': .99, 'evidence': 'final screen' if world.position == 2 else 'intermediate'}
        result = ReplayEngine(world.capture, world.act, world.home, model, lambda: [], log=lambda _: None).run('Reach requested final screen')
        self.assertTrue(result['completed'], result)
        self.assertEqual(calls, [('react', 0), ('judge_text', 1), ('react', 1), ('judge_text', 2)])
        self.assertEqual(result['metrics']['react_progress_checks'], 2)

    def test_react_subtask_does_not_compare_with_unrelated_recording_final(self):
        home, clock, world_clock = page('Home', 'Clock'), page('Alarms', 'World clock'), page('World clock', 'Dhaka')
        old = recording('Set alarm at 10:30 pm', [step(home, clock, 'Clock'), step(clock, page('Alarms', '10:30 PM'), 'Alarms')])
        world = World([home, clock, world_clock])
        calls = []
        def model(kind, system, payload, vision):
            calls.append(kind)
            if kind == 'plan':
                return {'action_id': 'stored', 'relation': 'related', 'shared_prefix': 1, 'confidence': .99}
            if kind == 'react':
                return {'action': 'tap', 'element_id': 1}
            return {'complete': True, 'confidence': .99, 'evidence': 'World clock displayed'}
        result = ReplayEngine(world.capture, world.act, world.home, model, lambda: [old], log=lambda _: None).run('Go to world clock')
        self.assertTrue(result['completed'], result)
        self.assertEqual(calls, ['plan', 'react', 'judge_text'])
        self.assertNotIn('replay_final_checks', result['metrics'])

    def test_missing_replay_target_still_blocks_wrong_tap(self):
        screens, action = self.gallery()
        screens[1] = page('Pictures', 'Delete')
        world = World(screens, actions=[action])
        engine = world.engine()
        engine.observe()
        hydrated_action, _, limit = engine.plan(action['source_task'])
        self.assertFalse(engine.replay(hydrated_action, limit))
        self.assertEqual(len(world.commands), 1)
        self.assertEqual(engine.failure, 'ambiguous_target')

    def test_backend_timeout_does_not_trigger_repeated_model_calls(self):
        world = World([page('Start', 'Next')])
        backend = Mock(side_effect=TimeoutError('NVIDIA request exceeded 60s'))
        engine = ReplayEngine(world.capture, world.act, world.home, backend, lambda: [], log=lambda _: None)
        result = engine.run('Unknown task')
        self.assertFalse(result['completed'])
        backend.assert_called_once()
        self.assertEqual(world.commands, [])

    def test_judge_payload_removes_coordinates_and_preserves_task_values(self):
        import json
        world = World([page('10:30 PM', 'Alarm saved')])
        backend = Mock(return_value={'complete': True, 'confidence': .99, 'evidence': 'Alarm saved at correct time'})
        engine = ReplayEngine(world.capture, world.act, world.home, backend, lambda: [], log=lambda _: None)
        self.assertTrue(engine.verify('Set alarm at 10:30 PM'))
        payload = json.loads(backend.call_args.args[2])
        self.assertEqual(payload['elements'][0]['content'], '10:30 PM')
        self.assertNotIn('bbox', payload['elements'][0])
        self.assertNotIn('ID', payload['elements'][0])

    def test_garbled_prefix_final_is_visually_judged_before_more_actions(self):
        screens, action = self.gallery()
        screens[1] = page('pictures', '7.00', 'ncodmnnieres')
        world = World(screens, actions=[action])
        calls = []
        def model(kind, system, payload, vision):
            import json
            calls.append(kind)
            self.assertEqual(kind, 'judge_vision')
            self.assertIsNotNone(vision)
            self.assertIn('expected_final_elements', json.loads(payload))
            return {'complete': True, 'confidence': .99, 'evidence': 'Gallery Pictures screen visible in screenshot'}
        engine = ReplayEngine(world.capture, world.act, world.home, model, lambda: world.actions, log=lambda _: None)
        result = engine.run('Go to Gallery app')
        self.assertTrue(result['completed'], result)
        self.assertEqual(calls, ['judge_vision'])
        self.assertEqual(len(world.commands), 1)
        self.assertEqual(result['react_steps'], 0)

    def test_failed_visual_judgement_uses_vision_for_react_action(self):
        screens, action = self.gallery()
        screens[1] = page('Garbled OCR 123')
        world = World(screens, actions=[action])
        calls = []
        def model(kind, system, payload, vision):
            calls.append(kind)
            self.assertIsNotNone(vision)
            if kind == 'judge_vision':
                return {'complete': len(calls) > 2, 'confidence': .99, 'evidence': 'Gallery visible' if len(calls) > 2 else 'Gallery not visible'}
            self.assertEqual(kind, 'react_vision')
            return {'action': 'tap', 'x': .5, 'y': .5}
        engine = ReplayEngine(world.capture, world.act, world.home, model, lambda: world.actions, log=lambda _: None)
        result = engine.run('Go to Gallery app')
        self.assertTrue(result['completed'], result)
        self.assertEqual(calls, ['judge_vision', 'react_vision', 'judge_vision'])

    def test_react_json_mismatch_uses_visual_judge(self):
        world = World([page('Garbled OCR')])
        backend = Mock(return_value={'complete': True, 'confidence': .99, 'evidence': 'Screenshot proves goal'})
        engine = ReplayEngine(world.capture, world.act, world.home, backend, lambda: [], log=lambda _: None)
        engine.react_final_screen = page('Gallery', 'Pictures', 'Albums')
        self.assertTrue(engine.check_react_progress('Open Gallery'))
        self.assertEqual(backend.call_args.args[0], 'judge_vision')
        self.assertIsNotNone(backend.call_args.args[3])

    def test_no_stored_final_uses_text_judge_then_explicit_vision_escalation(self):
        world = World([page('OCR labels')])
        backend = Mock(side_effect=[{'complete': False, 'confidence': .99, 'need_vision': True},
                                   {'complete': True, 'confidence': .99, 'evidence': 'Visual proof'}])
        engine = ReplayEngine(world.capture, world.act, world.home, backend, lambda: [], log=lambda _: None)
        self.assertTrue(engine.check_react_progress('Unrecorded goal'))
        self.assertEqual([c.args[0] for c in backend.call_args_list], ['judge_text', 'judge_vision'])


class BoundedWaitTests(unittest.TestCase):
    def test_slow_operation_reports_waiting_and_completes(self):
        release = threading.Event()
        logs = []
        def operation():
            release.wait(.02)
            return 'result'
        try:
            result = run_with_progress('test remote read', operation, timeout=.2, interval=.005, log=logs.append)
        finally:
            release.set()
        self.assertEqual(result, 'result')
        self.assertTrue(any('[WAIT]' in line for line in logs))
        self.assertIn('[WAIT-END]', logs[-1])

    def test_timeout_returns_without_waiting_for_late_worker(self):
        release = threading.Event()
        logs = []
        try:
            with self.assertRaises(TimeoutError):
                run_with_progress('blocked read', lambda: release.wait(), timeout=.025, interval=.01, log=logs.append)
        finally:
            release.set()
        self.assertIn('[TIMEOUT]', logs[-1])

    def test_worker_exception_is_reported_and_propagated(self):
        logs = []
        def fail():
            raise ValueError('bad response')
        with self.assertRaises(ValueError):
            run_with_progress('database', fail, timeout=.2, log=logs.append)
        self.assertIn('failed after', logs[-1])

    def test_nvidia_client_disables_sdk_retries(self):
        constructor = Mock(return_value=object())
        ns = {'_client': None, 'OpenAI': constructor, 'os': os,
              'config': SimpleNamespace(NVIDIA_BASE_URL='https://test.invalid', NVIDIA_API_KEY='test')}
        load_functions('nvidia_llm_bridge.py', {'_get_client'}, ns)
        with patch.dict(os.environ, {'NVIDIA_REQUEST_TIMEOUT_SEC': '12'}):
            ns['_get_client']()
        self.assertEqual(constructor.call_args.kwargs['timeout'], 12)
        self.assertEqual(constructor.call_args.kwargs['max_retries'], 0)

    def test_nvidia_default_timeout_is_two_hundred_seconds(self):
        constructor = Mock(return_value=object())
        ns = {'_client': None, 'OpenAI': constructor, 'os': os,
              'config': SimpleNamespace(NVIDIA_BASE_URL='https://test.invalid', NVIDIA_API_KEY='test')}
        load_functions('nvidia_llm_bridge.py', {'_get_client'}, ns)
        with patch.dict(os.environ, {}, clear=True):
            ns['_get_client']()
        self.assertEqual(constructor.call_args.kwargs['timeout'], 200)

    def test_nvidia_request_receives_real_timeout(self):
        create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))]))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        ns = {'List': list, 'Any': object, '_get_client': lambda: client, 'config': SimpleNamespace(NVIDIA_MODEL='test')}
        load_functions('nvidia_llm_bridge.py', {'_call_sync'}, ns)
        ns['_call_sync'](system_prompt='system', user_prompt='task', images_b64=[], max_tokens=10, timeout=7)
        self.assertEqual(create.call_args.kwargs['timeout'], 7)

    def test_png_image_is_sent_with_png_mime_and_glm_low_reasoning(self):
        import base64
        create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))]))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        ns = {'List': list, 'Any': object, '_get_client': lambda: client, 'config': SimpleNamespace(NVIDIA_MODEL='z-ai/glm-5.3-flash')}
        load_functions('nvidia_llm_bridge.py', {'_call_sync'}, ns)
        png = base64.b64encode(b'\x89PNG\r\n\x1a\nfixture').decode()
        with patch.dict(os.environ, {}, clear=True):
            ns['_call_sync'](system_prompt='judge', user_prompt='task', images_b64=[png], max_tokens=1024, timeout=200)
        request = create.call_args.kwargs
        self.assertEqual(request['messages'][1]['content'][0]['image_url']['url'], 'data:image/png;base64,' + png)
        self.assertEqual(request['extra_body'], {'reasoning_effort': 'low'})
        self.assertEqual(request['timeout'], 200)

    def test_other_models_do_not_receive_glm_reasoning_setting(self):
        create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))]))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        ns = {'List': list, 'Any': object, '_get_client': lambda: client, 'config': SimpleNamespace(NVIDIA_MODEL='other-model')}
        load_functions('nvidia_llm_bridge.py', {'_call_sync'}, ns)
        ns['_call_sync'](system_prompt='judge', user_prompt='task', images_b64=[], max_tokens=1024)
        self.assertNotIn('extra_body', create.call_args.kwargs)

    def test_async_bridge_propagates_timeout_into_worker(self):
        tree = ast.parse(Path('nvidia_llm_bridge.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NvidiaBridge')
        from typing import List, Optional, Dict, Any
        sync = Mock(return_value='{}')
        ns = {'asyncio': asyncio, 'json': __import__('json'), 'List': List, 'Optional': Optional, 'Dict': Dict, 'Any': Any, '_call_sync': sync}
        load_functions('nvidia_llm_bridge.py', {'_parse_json_response'}, ns)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), 'nvidia_llm_bridge.py', 'exec'), ns)
        bridge = ns['NvidiaBridge']()
        asyncio.run(bridge.call_json('system', 'task', timeout=9))
        self.assertEqual(sync.call_args.kwargs['timeout'], 9)

    def test_adb_command_timeout_returns_failure(self):
        ns = {'os': os, '_resolve_adb': lambda: 'adb', 'subprocess': SimpleNamespace(run=Mock(side_effect=subprocess.TimeoutExpired('adb', 30)), TimeoutExpired=subprocess.TimeoutExpired, PIPE=subprocess.PIPE)}
        load_functions('tool/adb_tools.py', {'_adb'}, ns)
        with patch('builtins.print'):
            self.assertEqual(ns['_adb']('adb devices'), 'ERROR')


if __name__ == '__main__':
    unittest.main()
