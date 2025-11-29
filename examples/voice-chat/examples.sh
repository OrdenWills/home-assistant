#!/bin/bash

export CKPT=/Users/paulabartabajo/src/liquid/leap-examples/examples/news-reader/LFM2-Audio-1.5B-GGUF
export INPUT_WAV=/Users/paulabartabajo/src/liquid/leap-examples/examples/news-reader/audio-samples/barackobamafederalplaza.mp3
export OUTPUT_WAV=/Users/paulabartabajo/src/liquid/leap-examples/examples/audio-transcription-cli/audio-samples/output.mp3
export PLATFORM=macos-arm64

# Audio to Speech Recognition (ASR)   
./lfm2-audio-macos-arm64/llama-lfm2-audio \
-m $CKPT/LFM2-Audio-1.5B-Q8_0.gguf \
--mmproj $CKPT/mmproj-audioencoder-LFM2-Audio-1.5B-Q8_0.gguf \
-mv $CKPT/audiodecoder-LFM2-Audio-1.5B-Q8_0.gguf \
-sys "Perform ASR." \
--audio $INPUT_WAV

# Text To Speech (TTS)
./lfm2-audio-macos-arm64/llama-lfm2-audio \
-m $CKPT/LFM2-Audio-1.5B-Q8_0.gguf \
--mmproj $CKPT/mmproj-audioencoder-LFM2-Audio-1.5B-Q8_0.gguf \
-mv $CKPT/audiodecoder-LFM2-Audio-1.5B-Q8_0.gguf \
-sys "Perform TTS." \
-p "My name is Pau Labarta Bajo and I love AI" \
--output $OUTPUT_WAV

# Text To Speech (TTS)
./lfm2-audio-macos-arm64/llama-lfm2-audio \
-m $CKPT/LFM2-Audio-1.5B-Q8_0.gguf \
--mmproj $CKPT/mmproj-audioencoder-LFM2-Audio-1.5B-Q8_0.gguf \
-mv $CKPT/audiodecoder-LFM2-Audio-1.5B-Q8_0.gguf \
-sys "Perform TTS.
Use the following voice: A male speaker delivers a very expressive and animated speech, with a low-pitch voice and a slightly close-sounding tone. The recording carries a slight background noise." \
-p "What is your name?" \
--output $OUTPUT_WAV