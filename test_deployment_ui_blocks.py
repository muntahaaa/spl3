import unittest
import gradio as gr
from deployment_ui_blocks import DeploymentBlocks

class DeploymentStreamTests(unittest.TestCase):
    def blocks(self):
        def run_all():
            yield "report", []
        with DeploymentBlocks() as blocks:
            button=gr.Button();report=gr.Markdown();table=gr.Dataframe()
            button.click(run_all,outputs=[report,table])
        return blocks

    def test_stream_uses_root_replacement_and_final_has_string_report(self):
        blocks=self.blocks()
        first=blocks.handle_streaming_diffs(0,["Starting",{"data":[]}],"session",1,False)
        self.assertIsInstance(first[0],str)
        patches=blocks.handle_streaming_diffs(0,["### Report",{"data":[["case","Passed"]]}],"session",1,False)
        self.assertEqual(patches[0],[["replace",[],"### Report"]])
        self.assertEqual(patches[1][0][1],[])
        final=blocks.handle_streaming_diffs(0,[None,None],"session",1,True)
        self.assertEqual(final[0],"### Report")
        self.assertEqual(final[1]["data"],[["case","Passed"]])

    def test_previous_table_snapshot_is_not_mutated(self):
        blocks=self.blocks();data=["Report",{"data":[["Pending"]]}]
        blocks.handle_streaming_diffs(0,data,"session",2,False)
        data[1]["data"][0][0]="Passed"
        final=blocks.handle_streaming_diffs(0,[None,None],"session",2,True)
        self.assertEqual(final[1]["data"],[["Pending"]])
