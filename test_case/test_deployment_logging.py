"""UI log streaming and explicit parser callback regressions; no live services."""
import json
import queue
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from replay_engine import ReplayEngine
from test_case.test_replay_engine import World
from test_case import test_replay_engine as fixtures
from test_case.test_replay_integration import load_functions


class ProgressLoggingTests(unittest.TestCase):
    def test_replay_logs_matching_actions_completion_without_model_calls(self):
        screens, action = fixtures.ReplayTests().gallery()
        world = World(screens, actions=[action], start=1)
        logs = []
        engine = ReplayEngine(world.capture, world.act, world.home, world.model, lambda: world.actions, log=logs.append)
        result = engine.run(action['source_task'])
        text = '\n'.join(logs)
        self.assertTrue(result['completed'])
        for stage in ('[START]', '[SCREEN]', '[CATALOG]', '[PLAN]', '[ALIGN]', '[REPLAY]', '[ACTION]', '[VERIFY]', '[FINISH]'):
            self.assertIn(stage, text)
        self.assertIn('Stored step 2: similarity=', text)
        self.assertIn('Planner call skipped', text)
        self.assertEqual(world.calls, [])

    def test_model_start_response_and_failure_are_visible(self):
        logs = []
        engine = ReplayEngine(lambda: None, lambda _: True, lambda: True,
                              lambda *args: {'action': 'done', 'reason': 'visible goal'}, lambda: [], log=logs.append)
        engine.ask('react', 'system', {'task': 'test'})
        self.assertIn('Starting react', logs[0])
        self.assertIn('text only (no image)', logs[0])
        self.assertIn('returned in', logs[-1])
        engine.model_fn = Mock(side_effect=RuntimeError('service unavailable'))
        with self.assertRaises(RuntimeError):
            engine.ask('react', 'system', {})
        self.assertIn('failed after', logs[-1])

    def test_ui_streams_worker_messages_and_final_result(self):
        def runner(**kwargs):
            kwargs['log_callback']('[PLAN] Exact match')
            kwargs['log_callback']('[REPLAY] Stored step 2')
            return {'status': 'completed', 'completed': True, 'close_actions': []}
        ns = {'gr': SimpleNamespace(update=lambda **kwargs: kwargs), 'threading': threading,
              'time': time, 'json': json, 'run_high_level_task': runner}
        load_functions('ui.py', {'_run_high_level'}, ns)
        with patch('builtins.print'):
            updates = list(ns['_run_high_level']('Task', 'device'))
        self.assertIn('[UI] Deployment queued', updates[0][0])
        self.assertIn('[PLAN] Exact match', updates[-1][0])
        self.assertIn('[REPLAY] Stored step 2', updates[-1][0])
        self.assertTrue(json.loads(updates[-1][1])['completed'])
        self.assertIn('Running:', updates[1][1])

    def test_ui_wait_heartbeat_preserves_event_history(self):
        real_queue = queue.Queue
        class QueueWithOnePause:
            def __init__(self):
                self.inner = real_queue()
                self.count = 0
            def put(self, value):
                self.inner.put(value)
            def get(self, timeout=None):
                self.count += 1
                if self.count == 2:
                    raise queue.Empty
                return self.inner.get(timeout=timeout)
        def runner(**kwargs):
            kwargs['log_callback']('[MODEL] Waiting for planner')
            return {'status': 'completed', 'close_actions': []}
        fake_thread = SimpleNamespace(Thread=lambda target, **kwargs: SimpleNamespace(start=target))
        ns = {'gr': SimpleNamespace(update=lambda **kwargs: kwargs), 'threading': fake_thread,
              'time': time, 'json': json, 'run_high_level_task': runner}
        load_functions('ui.py', {'_run_high_level'}, ns)
        with patch.object(queue, 'Queue', QueueWithOnePause), patch('builtins.print'):
            updates = list(ns['_run_high_level']('Task', 'device'))
        self.assertIn('[WAIT]', updates[1][0])
        self.assertIn('[UI] Deployment queued', updates[1][0])
        self.assertNotIn('[WAIT]', updates[-1][0])
        self.assertIn('[MODEL] Waiting for planner', updates[-1][0])

    def test_parser_run_forwards_callback_to_all_stages(self):
        logs = []
        ref = SimpleNamespace(delete=Mock())
        ns = {'submit_task': Mock(return_value='job'), 'wait_for_result': Mock(return_value={'elements': []}),
              'display_result': Mock(return_value='result.json'), '_ensure_firebase_initialized': Mock(),
              'db': SimpleNamespace(reference=lambda _: ref)}
        load_functions('OmniParser/client.py', {'run'}, ns)
        self.assertEqual(ns['run']('image', log_callback=logs.append), 'result.json')
        for name in ('submit_task', 'wait_for_result', 'display_result'):
            self.assertEqual(ns[name].call_args.kwargs['log_callback'], logs.append)
        ref.delete.assert_called_once()

    def test_parser_pending_status_reaches_callback(self):
        logs = []
        ref = SimpleNamespace(get=Mock(side_effect=[None, {'status': 'done'}]))
        ns = {'time': SimpleNamespace(time=Mock(side_effect=[0, 0, 1, 11, 11, 11, 12]), sleep=Mock()),
              '_ensure_firebase_initialized': Mock(), 'db': SimpleNamespace(reference=lambda _: ref)}
        load_functions('OmniParser/client.py', {'wait_for_result'}, ns)
        result = ns['wait_for_result']('job', log_callback=logs.append)
        self.assertEqual(result['status'], 'done')
        self.assertTrue(any('status=pending' in line for line in logs))
        self.assertTrue(any('Result received' in line for line in logs))


if __name__ == '__main__':
    unittest.main()
