import asyncio
import threading
import time
import tempfile
from pathlib import Path
import unittest
from deployment_control import Assistance, answer_request
from deployment_hierarchy import parse_hierarchy
from deployment_artifacts import DeploymentImages
from test_case.test_replay_engine import World, page

class DeploymentStrategyTests(unittest.TestCase):
    def test_hierarchy_bounds_ids_and_blank_search_field(self):
        xml='<hierarchy><node text="Clock" bounds="[10,20][110,220]" resource-id="launcher:clock"/><node text="" class="android.widget.EditText" resource-id="launcher:search" bounds="[0,300][200,400]"/></hierarchy>'
        result=parse_hierarchy(xml,200,400)
        self.assertEqual(result[0]['bbox'],[.05,.05,.55,.55])
        self.assertEqual(result[0]['resource_id'],'launcher:clock')
        self.assertEqual(result[1]['type'],'input')
        self.assertEqual(result[1]['content'],'Search')

    def test_assistance_waits_for_explicit_answer_and_preserves_text(self):
        control=Assistance(); result=[]
        thread=threading.Thread(target=lambda: result.append(control.request({'missing':'Clock'})))
        thread.start()
        for _ in range(50):
            if control.pending: break
            time.sleep(.01)
        self.assertEqual(result,[])
        self.assertIn('Provide more information',answer_request(control.token,''))
        answer_request(control.token,'Clock is in the Tools folder')
        thread.join(1)
        self.assertEqual(result,[{'skip':False,'info':'Clock is in the Tools folder'}])
        control.close()

    def test_skip_does_not_dispatch_react(self):
        world=World([page('Home')])
        engine=world.engine()
        engine.start_from_home=True
        engine.assistance_callback=lambda _: {'skip':True}
        engine.load_fn=lambda: []
        engine.open_app=lambda app,task: engine.request_assistance(task,'missing app')
        result=engine.run('Open missing app')
        self.assertEqual(result['status'],'skipped')
        self.assertFalse(result['completed'])
        self.assertEqual(world.calls,[])
        self.assertEqual(world.commands,[])

    def test_launcher_swipe_search_type_open_sequence(self):
        home=page('Home'); search=page('Search'); field=page('Search'); field['elements'][0]['type']='input'
        result=page('Clock')
        world=World([home,search,field,result,page('Timer')])
        engine=world.engine()
        self.assertTrue(engine.open_app('Clock','Start timer'))
        self.assertEqual([c['action'] for c in world.commands],['swipe','tap','text','tap'])
        self.assertEqual(world.commands[2]['input_str'],'Clock')
        self.assertEqual(world.calls,[])

    def test_react_rejoins_remaining_stored_steps_without_action_planner(self):
        from test_case.test_replay_engine import step,recording
        source, final=page('Albums'),page('Albums','Camera')
        action=recording('Switch to Albums',[step(source,final,'Albums')])
        world=World([source,final])
        engine=world.engine()
        engine.react_rejoin=(action,1,0)
        engine.react_final_screen=final
        engine.verify=lambda task,prefer_vision=False: True
        self.assertTrue(engine.react('Switch to Albums',action))
        self.assertEqual(world.calls,[])
        self.assertEqual(len(world.commands),1)
        self.assertEqual(engine.metrics['replayed_steps'],1)

    def test_late_json_xml_cleanup_and_failure_retention(self):
        with tempfile.TemporaryDirectory() as folder:
            tracker=DeploymentImages(lambda _:None,folder)
            tracker.finish(True)
            for name in ('log/screenshots/deployment/current.xml','labeled_image/json_labeled_data/current.json'):
                path=Path(folder)/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('test')
                tracker.track(path)
                self.assertFalse(path.exists())

if __name__=='__main__': unittest.main()


class CaseDeletionTests(unittest.TestCase):
    def test_deletion_preserves_shared_pages_and_elements(self):
        from types import SimpleNamespace
        from deployment_cases import delete_case
        class Rows(list):
            def single(self): return self[0] if self else None
            def consume(self): return None
        calls=[]
        class Tx:
            def run(self, query, **params):
                calls.append((query,params))
                if "RETURN a" in query and "<>" not in query:
                    return Rows([{"a":{"action_id":"delete-me"}}])
                if "RETURN a" in query:
                    return Rows([{"a":{"element_sequence":[{"element_id":"shared-element","source":{"page_id":"shared-page"}}]}}])
                if "AS source" in query and "<>" in query:
                    return Rows([{ "element":"shared-element", "source":"shared-page", "destination":None}])
                if "AS source" in query:
                    return Rows([{"element":"unique-element","source":"unique-page","destination":"shared-page"},
                                 {"element":"shared-element","source":"shared-page","destination":None}])
                if "AS count" in query: return Rows([{"count":0}])
                if "collect(e.element_id)" in query:
                    return Rows([{"p":{"page_id":"unique-page"},"elements":["unique-element"]}])
                return Rows()
        class Session:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def execute_write(self,fn): return fn(Tx())
        db=SimpleNamespace(driver=SimpleNamespace(session=lambda **kw:Session()),database="test")
        result=delete_case(db,"delete-me")
        self.assertEqual(result['pages'],['unique-page'])
        self.assertEqual(result['elements'],['unique-element'])
        self.assertEqual(result['status'],'deleted')
        for query,params in calls:
            if "DETACH DELETE" in query:
                self.assertNotIn('shared-page',params.get('ids',[]))
                self.assertNotIn('shared-element',params.get('ids',[]))
