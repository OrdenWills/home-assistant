import time

import torch
import torchaudio
from pathlib import Path
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState, LFMModality

from .modal_infra import (
    get_docker_image,
    get_modal_app,
    get_retries,
    get_secrets,
    get_volume,
)
from .audio_file_manager import AudioFileManager
from .audio_recorder import AudioRecorder
from .audio_player import AudioPlayer


app = get_modal_app("voice-chat-example")
image = get_docker_image()
volume = get_volume("voice-chat-example-volume")


# TODOs
# - Add modal decorator
# - Attach volume
# - Run
# @app.function(
#     image=image,
#     gpu="L40S",
#     volumes={
#         "/sessions": volume,
#     },
#     secrets=get_secrets(),
#     timeout=1 * 60 * 60,
#     retries=get_retries(max_retries=1),
#     max_inputs=1,  # Ensure we get a fresh container on retry
# )
# def run(session_id: str) -> str:

#     # Load models
#     HF_REPO = "LiquidAI/LFM2-Audio-1.5B"

#     processor = LFM2AudioProcessor.from_pretrained(HF_REPO).eval()
#     model = LFM2AudioModel.from_pretrained(HF_REPO).eval()

#     # Set up inputs for the model
#     chat = ChatState(processor)

#     # System prompt
#     chat.new_turn("system")
#     chat.add_text("Respond with interleaved text and audio.")
#     chat.end_turn()

#     # User prompt with audio input
#     chat.new_turn("user")
#     wav, sampling_rate = torchaudio.load(f"/sessions/{session_id}/question.wav")
#     chat.add_audio(wav, sampling_rate)
#     chat.end_turn()

#     chat.new_turn("assistant")

#     # Generate text and audio tokens.
#     text_out: list[torch.Tensor] = []
#     audio_out: list[torch.Tensor] = []
#     modality_out: list[LFMModality] = []
#     for t in model.generate_interleaved(**chat, max_new_tokens=512, audio_temperature=1.0, audio_top_k=4):
#         if t.numel() == 1:
#             print(processor.text.decode(t), end="", flush=True)
#             text_out.append(t)
#             modality_out.append(LFMModality.TEXT)
#         else:
#             audio_out.append(t)
#             modality_out.append(LFMModality.AUDIO_OUT)

#     # Detokenize audio, removing the last "end-of-audio" codes
#     # Mimi returns audio at 24kHz
#     mimi_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
#     with torch.no_grad():
#         waveform = processor.mimi.decode(mimi_codes)[0]
    
#     # Save the generated audio file
#     answer_path = f"/sessions/{session_id}/answer1.wav"
#     torchaudio.save(answer_path, waveform.cpu(), 24_000)
    
#     # Return the path to the generated audio file (remove /sessions prefix)
#     return answer_path.replace("/sessions", "")

def generate_session_id() -> str:
    """
    Generate a unique session ID based on the current timestamp.
    """
    import datetime
    now = datetime.datetime.now()
    session_id = now.strftime("%Y%m%d_%H%M%S")
    return session_id

def record() -> Path:
    """
    Records from the microphone and saves to a local wav file.

    Returns:
        Path to the saved wav file.
    """
    # import numpy as np
    
    recorder = AudioRecorder(sample_rate=16000, channels=1)

    print("Starting recording... Speak now!")
    
    # Start recording with auto-stop on silence and rich audio bar
    recorder.start_recording(
        silence_duration=2.0,    # Stop after 2 seconds of silence
        silence_threshold=0.02,  # Higher threshold = less sensitive to background noise
        show_audio_bar=True      # Enable rich audio visualization
    )

    # Wait for recording to finish automatically
    while recorder.is_recording:
        time.sleep(0.1)

    print("\nRecording finished, processing...")

    # Get the recorded audio
    audio_data = recorder.stop_recording()
    
    print(f"Audio data length: {len(audio_data)} samples")

    # Save it
    filename = "user_recording.wav"
    recorder.save_to_file(filename, audio_data)
    recorder.cleanup()
    
    # Return absolute path to the file
    return Path(filename).resolve()

@app.local_entrypoint()
def local_entrypoint():

    # generate a unique session id
    session_id = generate_session_id()

    # initialize audio file manager and push audio file to modal volume
    audio_manager = AudioFileManager(
        volume_name="voice-chat-example-volume",
        session_id=session_id
    )

    # record user question
    audio_file = record()

    # upload audio file (automatically goes to session directory)
    audio_manager.upload(str(audio_file))
    
    # run remote function and get the generated audio file path
    # remote_audio_path = run.remote(session_id)
    import modal
    remote_audio_path = modal.Function.from_name("voice-chat-example", "get_model_response").remote(session_id)
    print(f"Audio generated at: {remote_audio_path}")
    
    # download the generated answer
    local_answer_path = f"answer_{session_id}.wav"
    
    audio_manager.download("answer1.wav", local_answer_path)
    
    print(f"Generated audio saved to: {local_answer_path}")
    
    # Play the generated audio
    player = AudioPlayer()
    player.play(local_answer_path)
    player.cleanup()

if __name__ == "__main__":

    local_entrypoint()

