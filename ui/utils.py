import pandas as pd

def instruction_clips_to_playlist(instruction_clips):
    playlist = []
    for idx, step in enumerate(instruction_clips):
        if step.get("clips"):
            clip = step["clips"][0] # Take the top clip for each instruction step
            playlist.append({
                "step": idx + 1,
                "instruction": step["instruction"][:50] + "..." if len(step["instruction"]) > 50 else step["instruction"],
                "video_path": clip["video_path"],
                "start": clip["start_sec"],
                "score": round(clip["similarity_score"], 2)
            })
    return playlist

def playlist_to_df(playlist=None):
    if playlist is None or len(playlist) == 0:
        return pd.DataFrame(columns=["Step", "Instruction", "Start (s)", "Score"])
    
    return pd.DataFrame([
        {
            "Step": e["step"],
            "Instruction": e["instruction"],
            "Start (s)": e["start"],
            "Score": e["score"],
        }
        for e in playlist
    ])