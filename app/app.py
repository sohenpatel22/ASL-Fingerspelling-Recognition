from __future__ import annotations

import os
import tempfile
from functools import lru_cache

import gradio as gr

from asl.checkpoint import resolve_checkpoint
from asl.infer import Predictor
from asl.postprocess import MODES, enhance_text_with_llm
from asl.video import NoHandsDetected


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    path = resolve_checkpoint()
    return Predictor.from_checkpoint(path, device="cpu", allow_unsafe=False)


def translate(video_path: str | None, mode: str, speak: bool):
    if not video_path:
        return "Please upload or record a video.", None
    try:
        pred = get_predictor().predict_video(video_path)
    except NoHandsDetected as err:
        return str(err), None
    text = enhance_text_with_llm(pred.text, mode)
    summary = f"{text}\n\n(raw: '{pred.text}' | confidence {pred.confidence:.2f} | "
    summary += f"hands detected in {pred.hand_detection_rate:.0%} of {pred.n_frames} frames)"

    audio_path = None
    if speak and text:
        try:
            from gtts import gTTS

            audio_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
            gTTS(text=text, lang="en").save(audio_path)
        except Exception:
            audio_path = None
    return summary, audio_path


with gr.Blocks(theme="ocean") as demo:
    gr.Markdown("# ASL Fingerspelling Translator")
    gr.Markdown("MediaPipe landmarks -> Conformer-Transformer -> text | by Sohen Patel")
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Capture ASL", height=400)
            mode = gr.Radio(list(MODES), value="None", label="LLM post-processing (needs GROQ_API_KEY)")
            speak = gr.Checkbox(value=True, label="Speak the result")
            submit = gr.Button("Translate signs", variant="primary")
        with gr.Column():
            text_output = gr.Textbox(label="Model prediction", lines=5)
            audio_output = gr.Audio(label="Audio output", autoplay=True)
    submit.click(translate, [video_input, mode, speak], [text_output, audio_output])

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
