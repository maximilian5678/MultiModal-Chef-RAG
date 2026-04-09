import gradio as gr
from ui.state import UIState
import config
from ui.utils import playlist_to_df
from ui.handlers import create_submit_handler, on_playlist_select

def build_ui(pipeline):
    with gr.Blocks(title="Chat Interaction UI", theme=gr.themes.Ocean()) as demo:

        state = gr.State(UIState())
        
        on_submit = create_submit_handler(pipeline)

        with gr.Row():

            with gr.Column(scale=3):

                with gr.Row():
                    gr.Markdown("## CookBook AI")


                with gr.Column():
                    gr.Textbox(
                        label="System Prompt",
                        value=config.SYSTEM_PROMPT,
                        interactive=False,
                        lines=3,
                    )

                    context_box = gr.Textbox(
                        label="Injected Context",
                        value="No context loaded.",
                        lines=5,
                        interactive=False,
                    )

                chatbot = gr.Chatbot(
                    value=[
                        {"role": "assistant", "content": "Welcome. How can I assist you today?"}
                    ],
                    height=500,
                    type='messages',
                )

                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Type your message here...",
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("Send", scale=1)

            with gr.Column(scale=2):

                with gr.Column(elem_classes="panel"):
                    with gr.Row():
                        gr.Markdown("### 🎬 Video Exploration")
                        token_display = gr.Markdown("**Total Token Usage:** 0")

                with gr.Column():
                    
                    with gr.Column(scale=1):
                        video_player = gr.Video(
                            label="Video Player",
                            autoplay=True,
                        )

                        gr.Markdown("##### Video Playlist")

                        playlist = gr.Dataframe(
                            value=playlist_to_df(),
                            headers=["Step", "Instruction", "Start (s)", "Score"],
                            interactive=False,
                            row_count=(5, "dynamic"),
                        )

        send_btn.click(
            on_submit,
            inputs=[user_input, state],
            outputs=[chatbot, user_input, context_box, token_display, playlist, state]
        )

        user_input.submit(
            on_submit,
            inputs=[user_input, state],
            outputs=[chatbot, user_input, context_box, token_display, playlist, state]
        )
        playlist.select(
            on_playlist_select,
            inputs=[state],
            outputs=[video_player]
        )

    return demo