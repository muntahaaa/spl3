"""Offline checks for chain persistence, failure reporting, and run isolation."""
import asyncio
from contextvars import ContextVar
import json
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import unittest
from unittest.mock import AsyncMock, Mock, patch
from test_replay_integration import load_functions


class ChainConsistencyTests(unittest.TestCase):
    def test_per_stage_model_override_does_not_use_glm_options_for_other_models(self):
        from typing import Optional
        create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))]))
        ns = {'List': List, 'Optional': Optional, 'Any': Any,
              '_get_client': lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
              'config': SimpleNamespace(NVIDIA_MODEL='z-ai/glm-5.3-flash')}
        load_functions('nvidia_llm_bridge.py', {'_call_sync'}, ns)
        ns['_call_sync'](system_prompt='system', user_prompt='prompt', images_b64=[], max_tokens=1536, model_name='alternate-vision-model')
        self.assertEqual(create.call_args.kwargs['model'], 'alternate-vision-model')
        self.assertNotIn('extra_body', create.call_args.kwargs)

    def test_visual_recovery_keeps_images(self):
        response = {k: 'Exact evidence' for k in ('context', 'user_intent', 'state_change', 'task_relation', 'source_page_enhanced_desc', 'element_enhanced_desc', 'target_page_enhanced_desc')}
        call = AsyncMock(side_effect=[TimeoutError('primary'), response])
        ns = {'_call_reasoning': call, 'wait_for_llm_slot': AsyncMock(), '_TRIPLET_SYSTEM': 'system'}
        load_functions('chain_understand.py', {'_reason_triplet', '_recover_timed_out_triplet', '_validate_triplet_result'}, ns)
        with patch('builtins.print'):
            result, mode = asyncio.run(ns['_reason_triplet']('prompt', ['large-image'], AsyncMock(return_value=['small-image'])))
        self.assertEqual(mode, 'vision_recovery')
        self.assertEqual(result['evidence_source'], mode)
        self.assertEqual(call.await_args_list[1].args[2], ['small-image'])

    def test_access_denial_does_not_leak_between_runs(self):
        state = ContextVar('test-state', default=None)
        async def process(chain):
            self.assertFalse(ns['_access_denied']())
            ns['_mark_llm_access_denied'](RuntimeError('denied'))
            self.assertTrue(ns['_access_denied']())
            return chain
        ns = {'List': List, 'Dict': Dict, 'Any': Any, '_RUN_STATE': state, '_process_single_chain': process}
        load_functions('chain_understand.py', {'process_single_chain', '_access_denied', '_mark_llm_access_denied'}, ns)
        with patch('builtins.print'):
            asyncio.run(ns['process_single_chain']([1]))
            asyncio.run(ns['process_single_chain']([2]))
        self.assertIsNone(state.get())

    def test_failed_reasoning_job_is_error(self):
        update = Mock()
        ns = {'update_job': update, 'process_and_update_chain': AsyncMock(return_value=[{'reasoning_error': 'timeout'}])}
        load_functions('chain/chain_service.py', {'run_understand'}, ns)
        asyncio.run(ns['run_understand']('job', 'page'))
        self.assertEqual(update.call_args.args[1], 'error')
        self.assertFalse(any(c.args[1] == 'done' for c in update.call_args_list))

    def test_saved_reasoning_is_hydrated_for_evolution(self):
        import ast
        from pathlib import Path
        tree = ast.parse(Path('data/graph_db.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Neo4jDatabase')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_chain_from_start')
        reasoning = {'context': 'Clock', 'state_change': 'World clock opened'}
        records = [{'src': {'page_id': 's'}, 'tgt': {'page_id': 't', 'timestamp': 1}, 'lt': {'action_name': 'tap'}, 'e': {'element_id': 'e', 'reasoning': json.dumps(reasoning)}}]
        session = SimpleNamespace(run=lambda *a, **kw: records)
        class Session:
            def __enter__(self): return session
            def __exit__(self, *args): pass
        ns = {'List': List, 'Dict': Dict, 'Any': Any, 'json': json}
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'data/graph_db.py', 'exec'), ns)
        db = SimpleNamespace(driver=SimpleNamespace(session=lambda **kw: Session()), database='test')
        result = ns['get_chain_from_start'](db, 's')
        self.assertEqual(result[0]['reasoning'], reasoning)

    def pipeline(self, process, write_result=True):
        triplet = {'source_page': {'page_id': 's'}, 'target_page': {'page_id': 't'}, 'element': {'element_id': 'e'}, 'action': {'action_name': 'tap'}}
        ns = {'List': List, 'Dict': Dict, 'Any': Any, 'asyncio': asyncio, 'json': json, 'time': time,
              '_resolve_action_name': lambda t: 'tap', '_resolve_element': lambda t: t['element'],
              'process_triplet': process, 'merge_node_descriptions': AsyncMock(side_effect=lambda chain, task: chain),
              'update_node_in_db': Mock(return_value=write_result)}
        load_functions('chain_understand.py', {'_process_single_chain'}, ns)
        return ns, triplet

    def test_failed_triplet_never_persists_enriched_data(self):
        async def process(t):
            t['reasoning_error'] = 'timeout'
            return t
        ns, triplet = self.pipeline(process)
        with patch('builtins.print'), self.assertRaisesRegex(RuntimeError, 'Understanding incomplete'):
            asyncio.run(ns['_process_single_chain']([triplet]))
        ns['update_node_in_db'].assert_not_called()

    def test_failed_database_write_is_reported(self):
        async def process(t):
            t['reasoning'] = {'context': 'validated'}
            return t
        ns, triplet = self.pipeline(process, write_result=False)
        with patch('builtins.print'), self.assertRaisesRegex(RuntimeError, 'persistence failed'):
            asyncio.run(ns['_process_single_chain']([triplet]))

    def test_unreadable_images_use_recorded_text_without_crashing(self):
        async def thread(fn, *args):
            raise OSError('bad image')
        result = {key: 'Recorded description' for key in ('context', 'user_intent', 'state_change', 'task_relation', 'source_page_enhanced_desc', 'element_enhanced_desc', 'target_page_enhanced_desc')}
        reason = AsyncMock(return_value=(result, 'text'))
        ns = {'Dict': Dict, 'Any': Any, 'List': List, 'asyncio': SimpleNamespace(to_thread=thread, gather=asyncio.gather),
              'json': json, 'time': time, '_resolve_element': lambda t: t['element'], '_resolve_action_name': lambda t: 'tap',
              '_build_triplet_user_prompt': lambda **kw: 'Recorded transition', '_load_image_to_base64': Mock(),
              '_access_denied': lambda: False, 'wait_for_llm_slot': AsyncMock(), '_reason_triplet': reason}
        load_functions('chain_understand.py', {'process_triplet'}, ns)
        triplet = {'source_page': {'page_id': 's', 'raw_page_url': 'bad.png'}, 'target_page': {'page_id': 't', 'raw_page_url': 'bad.png'}, 'element': {'element_id': 'e'}}
        with patch('builtins.print'):
            processed = asyncio.run(ns['process_triplet'](triplet))
        self.assertNotIn('reasoning_error', processed)
        self.assertEqual(processed['reasoning_source'], 'text')
        self.assertEqual(reason.call_args.args[1], [])


class ModelJsonFormattingTests(unittest.TestCase):
    def test_preamble_and_fences_accept_one_complete_object(self):
        import json
        from test_replay_integration import load_functions
        namespace = {"json": json}
        load_functions("nvidia_llm_bridge.py", {"_parse_json_response"}, namespace)
        parse = namespace["_parse_json_response"]
        self.assertEqual(parse('Here is the result:\n```json\n{"context":"Clock"}\n```'), {"context": "Clock"})
        with self.assertRaises(json.JSONDecodeError):
            parse('{"context":')
        with self.assertRaises(ValueError):
            parse('{} {"other":1}')

    def test_llama_collage_preserves_both_screen_panels(self):
        import base64, io
        from PIL import Image, ImageDraw
        from test_replay_integration import load_functions
        namespace = {"base64": base64, "io": io, "Image": Image, "ImageDraw": ImageDraw}
        load_functions("chain_understand.py", {"_combine_triplet_images"}, namespace)
        images = []
        for color in ("red", "blue"):
            buffer = io.BytesIO()
            Image.new("RGB", (40, 80), color).save(buffer, format="PNG")
            images.append(base64.b64encode(buffer.getvalue()).decode())
        combined = namespace["_combine_triplet_images"](images)
        with Image.open(io.BytesIO(base64.b64decode(combined))) as image:
            self.assertEqual(image.size, (80, 112))
            self.assertGreater(image.getpixel((20, 70))[0], 200)
            self.assertGreater(image.getpixel((60, 70))[2], 200)


class LlamaRecoveryTests(unittest.TestCase):
    def test_format_recovery_does_not_resend_images(self):
        failure = ValueError("invalid JSON")
        failure.response_text = "Home screen to World clock after tapping Clock."
        response = {key: "Observed Clock transition" for key in ("context", "user_intent", "state_change", "task_relation", "source_page_enhanced_desc", "element_enhanced_desc", "target_page_enhanced_desc")}
        call = AsyncMock(side_effect=[failure, response])
        namespace = {"_call_reasoning": call, "wait_for_llm_slot": AsyncMock(), "_TRIPLET_SYSTEM": "JSON schema"}
        load_functions("chain_understand.py", {"_reason_triplet", "_recover_timed_out_triplet", "_validate_triplet_result"}, namespace)
        with patch("builtins.print"):
            result, mode = asyncio.run(namespace["_reason_triplet"]("Recorded task", ["screenshot"]))
        self.assertEqual(mode, "vision_format_recovery")
        self.assertEqual(call.await_count, 2)
        self.assertIsNone(call.await_args_list[1].args[2])
        self.assertIn(failure.response_text, call.await_args_list[1].args[1])

    def test_llama_vision_instructions_are_in_user_message(self):
        import base64
        create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]))
        namespace = {"_get_client": lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), "config": SimpleNamespace(NVIDIA_MODEL="meta/llama-3.2-11b-vision-instruct")}
        load_functions("nvidia_llm_bridge.py", {"_call_sync"}, namespace)
        namespace["_call_sync"](system_prompt="Required JSON schema", user_prompt="Inspect Clock", images_b64=[base64.b64encode(b"fake jpeg").decode()], max_tokens=1536)
        request = create.call_args.kwargs
        self.assertEqual(len(request["messages"]), 1)
        self.assertIn("Required JSON schema", request["messages"][0]["content"][-1]["text"])
        self.assertNotIn("extra_body", request)
