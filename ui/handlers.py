import gradio as gr
from pathlib import Path
from ui.state import UIState
from ui.utils import playlist_to_df, instruction_clips_to_playlist

def create_submit_handler(pipeline):
    """Factory function that creates a handler with pipeline in closure"""
    def on_submit(user_input, state: UIState):
        if not user_input.strip():
            return (
                state.chat_history,
                "",
                state.context,
                f"**Total Token Usage:** {state.token_usage}",
                playlist_to_df(state.playlist),
                state
            )

        instruction_clips, context, updated_history, used_tokens = pipeline.run_rag_pipeline(
            user_input,
            state.chat_history
        )

        state.chat_history = updated_history 
        state.context = context
        state.token_usage += used_tokens
        state.playlist = instruction_clips_to_playlist(instruction_clips)

        return (
            state.chat_history,
            "",
            state.context,
            f"**Total Token Usage:** {state.token_usage}",
            playlist_to_df(state.playlist),
            state
        )
    
    return on_submit

def on_playlist_select(evt: gr.SelectData, state: UIState):
    """Handler for when a user selects a row in the playlist table"""
    idx = evt.index

    if isinstance(idx, (list, tuple)):
        row = idx[0]
    else:
        row = idx # Extract the row index from the event

    if not state.playlist or row >= len(state.playlist):
        return None
    
    entry = state.playlist[int(row)]
    video_path = (Path("project") / entry["video_path"]).resolve()
    start = int(entry["start"])

    video_with_seek = f"{video_path}#t={start}"

    print(f"Selected video: {video_with_seek}")
    return str(video_with_seek)
