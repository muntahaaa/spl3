"""Use whole-value stream patches for deployment UI outputs."""
import copy
import gradio as gr


class DeploymentBlocks(gr.Blocks):
    def handle_streaming_diffs(self, fn_index, data, session_hash, run, final, simple_format=False):
        fn = self.fns[fn_index].fn
        if getattr(fn, "__name__", "") not in {"_run_high_level", "run_selected", "run_all"}:
            return super().handle_streaming_diffs(fn_index,data,session_hash,run,final,simple_format)
        if session_hash is None or run is None:
            return data
        first = run not in self.pending_diff_streams[session_hash]
        # Keep immutable snapshots: table rows are updated between generator yields.
        full = super().handle_streaming_diffs(fn_index,copy.deepcopy(data),session_hash,run,final,True)
        if first or final or simple_format:
            return full
        # The Gradio client supports replacement at the root path. No nested
        # indices or string-append operations can point into a stale UI value.
        return [[["replace", [], value]] for value in full]
