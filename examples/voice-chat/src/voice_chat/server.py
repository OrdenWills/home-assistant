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
from .model_loader import ModelLoader

app = get_modal_app("voice-chat-example")
image = get_docker_image()
sessions_volume = get_volume("voice-chat-example-volume")
model_volume = get_volume("models")

@app.function(
    image=image,
    gpu="L40S",
    volumes={
        "/sessions": sessions_volume,
        "/model_cache": model_volume,
    },
    secrets=get_secrets(),
    timeout=1 * 60 * 60,
    retries=get_retries(max_retries=1),
    max_inputs=1,  # Ensure we get a fresh container on retry
)
def get_model_response(session_id: str) -> str:
    """
    Run the LFM2-Audio-1.5B model to generate a voice response for the given session ID.
    
    Args:
        session_id: Unique session identifier for organizing files

    Returns:
        Path to the generated answer audio file (relative to /sessions)
    """
    processor, model = ModelLoader().load_models()

    # Set up inputs for the model
    chat = ChatState(processor)

    # System prompt
    chat.new_turn("system")
    chat.add_text("Respond with interleaved text and audio.")
    chat.end_turn()

    # User prompt with audio input
    chat.new_turn("user")
    wav, sampling_rate = torchaudio.load(f"/sessions/{session_id}/question.wav")
    chat.add_audio(wav, sampling_rate)
    chat.end_turn()

    chat.new_turn("assistant")

    # Generate text and audio tokens.
    text_out: list[torch.Tensor] = []
    audio_out: list[torch.Tensor] = []
    modality_out: list[LFMModality] = []
    for t in model.generate_interleaved(**chat, max_new_tokens=512, audio_temperature=1.0, audio_top_k=4):
        if t.numel() == 1:
            print(processor.text.decode(t), end="", flush=True)
            text_out.append(t)
            modality_out.append(LFMModality.TEXT)
        else:
            audio_out.append(t)
            modality_out.append(LFMModality.AUDIO_OUT)

    # Detokenize audio, removing the last "end-of-audio" codes
    # Mimi returns audio at 24kHz
    mimi_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
    with torch.no_grad():
        waveform = processor.mimi.decode(mimi_codes)[0]
    
    # Save the generated audio file
    answer_path = f"/sessions/{session_id}/answer1.wav"
    torchaudio.save(answer_path, waveform.cpu(), 24_000)
    
    # Return the path to the generated audio file (remove /sessions prefix)
    return answer_path.replace("/sessions", "")