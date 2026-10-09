"""Offline contracts for concise chain reasoning and avoided merge calls."""
import ast
import asyncio
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import unittest
from unittest.mock import AsyncMock, Mock, patch
from test_case.test_replay_integration import load_functions


class ChainFastPathTests(unittest.TestCase):
    def test_both_chain_bridges_force_low_reasoning_and_bounded_output(self):
        for filename in ('chain_understand.py', 'chain_evolve.py'):
            tree = ast.parse(Path(filename).read_text(encoding='utf-8'))
            assign = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'bridge' for t in n.targets))
            factory = Mock()
            exec(compile(ast.Module(body=[assign], type_ignores=[]), filename, 'exec'), {'NvidiaBridge': factory, 'os': os, 'config': SimpleNamespace(NVIDIA_MODEL='configured-model')})
            self.assertEqual(factory.call_args.kwargs['reasoning_effort'], 'low')
            self.assertEqual(factory.call_args.kwargs['max_tokens_json'], 3072 if filename == 'chain_evolve.py' else 1536)

    def merge(self, first, second):
        bridge = SimpleNamespace(call_text=AsyncMock(return_value='Merged exact labels and values'))
        wait = AsyncMock()
        async def call_reasoning(system, prompt, images=None, text=False):
            return await bridge.call_text(system_prompt=system, user_prompt=prompt)
        ns = {'List': List, 'Dict': Dict, 'Any': Any, '_access_denied': lambda: False, '_call_reasoning': call_reasoning, 'bridge': bridge,
              'wait_for_llm_slot': wait, 'time': time, '_MERGE_SYSTEM': 'brief merge',
              '_build_merge_user_prompt': lambda **kw: str(kw)}
        load_functions('chain_understand.py', {'merge_node_descriptions'}, ns)
        chain = [{'target_page': {'page_id': 'shared', 'description': first}},
                 {'source_page': {'page_id': 'shared', 'description': second}}]
        with patch('builtins.print'):
            result = asyncio.run(ns['merge_node_descriptions'](chain, 'task'))
        return result, bridge, wait

    def test_identical_description_merge_skips_model_and_rate_slot(self):
        result, bridge, wait = self.merge('Clock  7 AM', 'Clock 7 AM')
        bridge.call_text.assert_not_called()
        wait.assert_not_called()
        self.assertEqual(result[0]['target_page']['description'], result[1]['source_page']['description'])

    def test_empty_description_merge_preserves_existing_content_without_model(self):
        result, bridge, wait = self.merge('', 'Alarm saved at 7 AM')
        bridge.call_text.assert_not_called()
        self.assertEqual(result[0]['target_page']['description'], 'Alarm saved at 7 AM')

    def test_distinct_descriptions_still_receive_semantic_merge(self):
        result, bridge, wait = self.merge('Clock alarm list', 'Saved alarm at 7 AM')
        bridge.call_text.assert_awaited_once()
        wait.assert_awaited_once()
        self.assertEqual(result[0]['target_page']['description'], 'Merged exact labels and values')

    def test_explicit_low_reasoning_overrides_environment_max(self):
        create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))]))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        ns = {'List': List, 'Any': Any, 'Optional': Optional, '_get_client': lambda: client,
              'config': SimpleNamespace(NVIDIA_MODEL='z-ai/glm-5.3-flash')}
        load_functions('nvidia_llm_bridge.py', {'_call_sync'}, ns)
        with patch.dict(os.environ, {'NVIDIA_REASONING_EFFORT': 'max'}):
            ns['_call_sync'](system_prompt='system', user_prompt='task', images_b64=[], max_tokens=1536, reasoning_effort='low')
        self.assertEqual(create.call_args.kwargs['extra_body']['reasoning_effort'], 'low')

    def test_generation_requests_no_discarded_sequence(self):
        tree = ast.parse(Path('chain_evolve.py').read_text(encoding='utf-8'))
        assign = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_GEN_SYSTEM' for t in n.targets))
        prompt = ast.literal_eval(assign.value)
        self.assertIn('element_sequence (empty list', prompt)
        self.assertIn('code reconstructs exact recorded actions', prompt)


class TripletRecoveryTests(unittest.TestCase):
    def response(self):
        return {key: 'Recorded factual description' for key in ('context', 'user_intent', 'state_change', 'task_relation', 'source_page_enhanced_desc', 'element_enhanced_desc', 'target_page_enhanced_desc')}

    def helper(self, responses):
        bridge = SimpleNamespace(call_json=AsyncMock(side_effect=responses))
        wait = AsyncMock()
        ns = {'bridge': bridge, 'wait_for_llm_slot': wait, '_TRIPLET_SYSTEM': 'brief JSON', 'asyncio': asyncio, 'time': time}
        load_functions('chain_understand.py', {'_reason_triplet', '_call_reasoning', '_recover_timed_out_triplet', '_validate_triplet_result'}, ns)
        return ns['_reason_triplet'], bridge, wait

    def test_visual_timeout_recovers_once_using_recorded_text(self):
        helper, bridge, wait = self.helper([TimeoutError('read timed out'), self.response()])
        with patch('builtins.print'):
            result, mode = asyncio.run(helper('Recorded source/action/destination', ['image']))
        self.assertEqual(mode, 'text_recovery')
        self.assertEqual(result['evidence_source'], 'text_recovery')
        self.assertEqual(bridge.call_json.await_count, 2)
        self.assertIsNone(bridge.call_json.await_args_list[1].kwargs['images_b64'])
        self.assertIn('do not invent visual details', bridge.call_json.await_args_list[1].kwargs['user_prompt'])
        wait.assert_awaited_once()

    def test_recovery_timeout_does_not_loop(self):
        helper, bridge, _ = self.helper([TimeoutError('vision timeout'), TimeoutError('text timeout')])
        with patch('builtins.print'), self.assertRaises(TimeoutError):
            asyncio.run(helper('prompt', ['image']))
        self.assertEqual(bridge.call_json.await_count, 2)

    def test_text_timeout_and_bad_response_are_not_retried(self):
        for response in (TimeoutError('timeout'), {'context': 'missing fields'}):
            helper, bridge, _ = self.helper([response])
            with self.assertRaises((TimeoutError, ValueError)):
                asyncio.run(helper('prompt', []))
            self.assertEqual(bridge.call_json.await_count, 1)

    def test_stream_collects_content_ignores_reasoning_and_closes(self):
        class Stream:
            def __init__(self): self.close = Mock()
            def __iter__(self):
                yield SimpleNamespace(choices=[])
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None, reasoning_content='private reasoning'), finish_reason=None)])
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='{"ok":'), finish_reason=None)])
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='true}'), finish_reason='stop')])
        stream = Stream()
        create = Mock(return_value=stream)
        ns = {'List': List, 'Optional': Optional, 'Any': Any,
              '_get_client': lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
              'config': SimpleNamespace(NVIDIA_MODEL='test-model')}
        load_functions('nvidia_llm_bridge.py', {'_call_sync'}, ns)
        with patch('builtins.print'):
            result = ns['_call_sync'](system_prompt='s', user_prompt='u', images_b64=[], max_tokens=1536, stream=True)
        self.assertEqual(result, '{"ok":true}')
        self.assertTrue(create.call_args.kwargs['stream'])
        stream.close.assert_called_once()

    def test_truncated_stream_is_rejected_and_closed(self):
        class Stream:
            def __init__(self): self.close = Mock()
            def __iter__(self):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='{"partial"'), finish_reason='length')])
        stream = Stream()
        ns = {'List': List, 'Optional': Optional, 'Any': Any,
              '_get_client': lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: stream))),
              'config': SimpleNamespace(NVIDIA_MODEL='test-model')}
        load_functions('nvidia_llm_bridge.py', {'_call_sync'}, ns)
        with patch('builtins.print'), self.assertRaisesRegex(ValueError, 'truncated'):
            ns['_call_sync'](system_prompt='s', user_prompt='u', images_b64=[], max_tokens=1536, stream=True)
        stream.close.assert_called_once()

class EvolveDescriptionTests(unittest.TestCase):
    def test_generation_describes_each_page_and_element_without_numeric_identity(self):
        tree=ast.parse(Path("chain_evolve.py").read_text(encoding="utf-8"))
        assign=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="_GEN_SYSTEM" for t in n.targets))
        prompt=ast.literal_eval(assign.value)
        for text in ("step_descriptions", "source_page_description", "element_description", "target_page_description", "not numeric values", "Ground all statements"):
            self.assertIn(text,prompt)
