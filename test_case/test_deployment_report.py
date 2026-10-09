import json
import unittest
from deployment_report import classify_outcome,stream_case_execution

class DeploymentReportTests(unittest.TestCase):
    def cases(self):
        return [{"action_id":str(i),"source_task":name} for i,name in enumerate(("Start timer","Open Clock","Reset Stopwatch"))]

    def test_combined_logs_marks_and_report(self):
        statuses=[{"completed":True,"completion_evidence":"Timer running"},{"status":"skipped","message":"Add new case"},{"status":"error","message":"Target missing"}]
        def runner(task,device,action_id):
            yield "[ACTION] Step 1", "Running", {"visible":False}, [], "", ""
            yield "[ACTION] Step 1\n[FINISH] Case finished", json.dumps(statuses[int(action_id)]), {"visible":False}, [], "", ""
        events=list(stream_case_execution(self.cases(),"device",runner,lambda **kw:kw))
        logs,outcome,popup,table,token,question,current,rows,report=events[-1]
        self.assertEqual([row[2] for row in rows],["✅ Passed","⏭ Skipped","❌ Failed"])
        self.assertIn("Start timer",logs)
        self.assertIn("Reset Stopwatch",logs)
        self.assertIn("Execution report",report)
        self.assertIn("Passed: **1**",report)
        self.assertIn("Failed: **1**",report)
        self.assertTrue(any("Case 2/3: Open Clock" in event[6] for event in events))

    def test_two_passed_cases_finish_with_an_atomic_table_snapshot(self):
        cases = self.cases()[:2]
        def runner(*args, **kwargs):
            yield "[FINISH] completed", json.dumps({"completed": True}), {"visible": False}, [], "", ""

        events = list(stream_case_execution(cases, "device", runner, lambda **kw: kw))
        final = events[-1]

        self.assertEqual(final[6], "Execution finished")
        self.assertEqual([row[2] for row in final[7]], ["✅ Passed", "✅ Passed"])
        self.assertIn("Passed: **2**", final[8])
        self.assertIn("Remaining: **0**", final[8])
        self.assertIsNot(final[7], events[-2][7])

    def test_exception_marks_failed_and_continues(self):
        calls=[]
        def runner(task,device,action_id):
            calls.append(action_id)
            if action_id=="0": raise RuntimeError("backend failure")
            yield "completed",json.dumps({"completed":True}),{},[],"",""
        final=list(stream_case_execution(self.cases(),"device",runner,lambda **kw:kw))[-1]
        self.assertEqual(calls,["0","1","2"])
        self.assertEqual(final[7][0][2],"❌ Failed")
        self.assertIn("backend failure",final[0])

    def test_assistance_token_and_visibility_survive_streaming(self):
        def runner(*args,**kwargs):
            yield "Waiting for user","Paused",{"visible":True},[],"run-token","Which control should be tapped?"
            yield "Skipped",json.dumps({"status":"skipped"}),{"visible":False},[],"",""
        events=list(stream_case_execution(self.cases()[:1],"device",runner,lambda **kw:kw))
        self.assertTrue(any(event[2].get("visible") and event[4]=="run-token" for event in events))
        self.assertTrue(any(event[5]=="Which control should be tapped?" for event in events))

    def test_empty_cases_and_bad_results(self):
        final=list(stream_case_execution([],"device",None,lambda **kw:kw))[-1]
        self.assertEqual(final[7],[])
        self.assertEqual(classify_outcome("not JSON")[0],"failed")
        self.assertEqual(classify_outcome('{"status":"skipped: add a case"}')[0],"skipped")

if __name__=="__main__": unittest.main()

class SuccessOutcomeTests(unittest.TestCase):
    def test_passed_outcome_exposes_actual_success_values(self):
        from deployment_report import outcome_with_success_values
        result={"completed":True,"completion_evidence":"Stopwatch reset to zero","steps_completed":5,"replayed_steps":5,"react_steps":0,"metrics":{"text_calls":0,"vision_calls":1},"returned_home":True,"image_cleanup":{"deleted":8,"errors":[]}}
        outcome=outcome_with_success_values(result,"Start stopwatch then reset")
        summary=json.loads(outcome)["success_summary"]
        self.assertEqual(summary["result"],"PASSED")
        self.assertEqual(summary["temporary_files_deleted"],8)
        self.assertTrue(summary["returned_home"])
        self.assertEqual(classify_outcome(outcome)[0],"completed")

    def test_missing_metrics_are_not_invented(self):
        from deployment_report import outcome_with_success_values
        summary=json.loads(outcome_with_success_values({"completed":True},"Task"))["success_summary"]
        self.assertEqual(summary["vision_model_calls"],"Not reported")

    def test_finished_stream_preserves_success_outcome(self):
        def runner(*args,**kwargs):
            yield "Done",json.dumps({"completed":True,"completion_evidence":"Albums visible"}),{},[],"",""
        events=list(stream_case_execution([{"action_id":"one","name":"Albums"}],"device",runner,lambda **kw:kw))
        self.assertIn("Albums visible",events[-1][1])
